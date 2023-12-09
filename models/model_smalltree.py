"""
Small TreeVAE model.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from models.networks import (get_decoder, MLP, Router, Dense)
from utils.model_utils import compute_posterior
from utils.training_utils import calc_aug_loss

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

class SmallTreeVAE(tf.keras.Model):
    def __init__(self, depth, bottom_up, **kwargs):
        super(SmallTreeVAE, self).__init__(name="SmallTreeVAE")
        self.kwargs = kwargs
        encoded_sizes = self.kwargs['latent_dim']
        hidden_layers = self.kwargs['mlp_layers']
        self.encoded_size = encoded_sizes[-depth-1]
        self.hidden_layer = hidden_layers[-depth]
        self.inp_shape = self.kwargs['inp_shape']
        self.activation = self.kwargs['activation']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']


        self.denses = [Dense(self.encoded_size) for i in range(2)]

        self.transformations = [MLP(self.encoded_size, self.hidden_layer) for i in range(2)]
        self.bottom_up = bottom_up[:-depth]
        self.bottom_up_q = bottom_up[-depth]

        self.decision = Router(hidden_units=self.hidden_layer)
        self.decision_q = Router(hidden_units=self.hidden_layer)
        self.decoders = [get_decoder(architecture=self.kwargs['encoder'], hidden_layer=hidden_layers[0],
                              input_shape=self.inp_shape, activation=self.activation) for i in range(2)]


    def call(self, inputs, training=True):
        epsilon = tf.keras.backend.epsilon()
        x, z_parent, p = inputs
        x = tf.keras.layers.Flatten()(x)
        
        # Bottom up pass until subtree
        d = x
        for i in range(0, len(self.bottom_up)):
            d, _, _ = self.bottom_up[i](d, False)


        # computation of decisions
        router_q = self.decision_q
        d_q, _, _ = self.bottom_up_q(d, False)
        
        prob_child_left = tf.squeeze(self.decision(z_parent, training))
        prob_child_left_q = tf.squeeze(router_q(tf.stop_gradient(d_q), training))
        leaves_prob = [prob_child_left_q,(1 - prob_child_left_q)]

        kl_decisions = prob_child_left_q * tf.math.log(epsilon + prob_child_left_q / (prob_child_left + epsilon)) +\
                        (1 - prob_child_left_q) * tf.math.log(epsilon + (1 - prob_child_left_q) /
                                                                (1 - prob_child_left + epsilon))
        kl_decisions = tf.math.reduce_mean(p * kl_decisions)
        

        # contrastive loss
        aug_decisions_loss = tf.zeros(1)
        if self.augment and 'simple' not in self.augmentation_method and training is True:
            aug_losses = self.augmentation_method
            aug_decisions_loss += calc_aug_loss(prob_parent=p, prob_router=prob_child_left_q, augmentation_methods=aug_losses)



        reconstructions = []
        # iterate for both children
        kl_nodes = 0
        for i in range(2):
            # compute posterior parameters
            z_mu_q_hat, z_sigma_q_hat = self.denses[i](tf.stop_gradient(d))
            _, z_mu_p, z_sigma_p = self.transformations[i](z_parent, training)
            z_p = tfd.MultivariateNormalDiag(loc=z_mu_p, scale_diag=tf.math.sqrt(z_sigma_p))
            z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

            # compute sample z using mu_q and sigma_q
            z = tfd.MultivariateNormalDiag(loc=z_mu_q, scale_diag=tf.math.sqrt(z_sigma_q))
            mu_var_q = tf.concat([z_mu_q, tf.math.sqrt(z_sigma_q)], axis=1)
            l = z_mu_q.shape[1]
            z_q_dist = tfp.layers.DistributionLambda(
                lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :l], scale_diag=theta[:, l:]))
            z_sample = z_q_dist(mu_var_q)


            # compute KL node
            kl_node = tf.math.reduce_mean(leaves_prob[i] * p * tfd.kl_divergence(z, z_p))
            kl_nodes += kl_node


            reconstructions.append(self.decoders[i](z_sample))

        kl_nodes_loss = tf.clip_by_value(kl_nodes, clip_value_min=-10, clip_value_max=10E10)

        # probability of falling in each leaf, should be of shape [batch_size, num_clusters]
        p_c_z = tf.concat([tf.expand_dims(leaves_prob[i], -1) for i in range(len(leaves_prob))], axis=-1)
        p_c_z = tf.cast(p_c_z, tf.float64)

        rec_loss = self.loss(x, reconstructions, leaves_prob)

        return {
            'rec_loss': rec_loss,
            'weights': leaves_prob,
            'kl_decisions': kl_decisions,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
        }


    def compile(self, optimizer, loss, metric, alpha=0.):
        super().compile(optimizer)
        self.loss = loss
        self.metric = metric
        self.alpha = tf.Variable(alpha, trainable=False)
        # Metrics
        self.loss_value = tfk.metrics.Mean(name="loss_value")
        self.rec_loss = tfk.metrics.Mean(name="rec_loss")
        self.kl_decisions = tfk.metrics.Mean(name="kl_decisions")
        self.kl_nodes = tfk.metrics.Mean(name="kl_nodes")
        self.nmi = tfk.metrics.Mean(name="nmi")
        self.perc_samples = tfk.metrics.Mean(name="perc_left")
        self.aug_decisions = tfk.metrics.Mean(name='aug_decisions')

    def train_step(self, data):
        inputs, y = data
        with tf.GradientTape() as tape:
            output = self(inputs, training=True)
            rec_loss = output['rec_loss']
            loss_value = rec_loss + self.alpha * (output['kl_nodes'] + output['kl_decisions']) + output['aug_decisions']
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        metric = self.metric(y, output['p_c_z'])
        
        # Monitor splits
        self.perc_samples.update_state(1 - tf.reduce_mean(tf.cast(tf.argmax(output['p_c_z'], axis=-1), "float")))


        self.loss_value.update_state(loss_value)
        self.rec_loss.update_state(rec_loss)
        self.kl_decisions.update_state(output['kl_decisions'])
        self.kl_nodes.update_state(output['kl_nodes'])
        self.aug_decisions.update_state(output['aug_decisions'])
        self.nmi.update_state(metric)

        return {"loss_value": self.loss_value.result(), "rec_loss": self.rec_loss.result(),
                "kl_decisions": self.kl_decisions.result(), "aug_decisions": self.aug_decisions.result(), "kl_nodes": self.kl_nodes.result(),
                "nmi": self.nmi.result(), 'perc_left': self.perc_samples.result(), 'alpha': self.alpha}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_value, self.rec_loss, self.kl_decisions, self.kl_nodes, self.nmi, self.perc_samples, self.aug_decisions]

    def test_step(self, data):
        # Unpack the data
        inputs, y = data
        # Compute predictions
        output = self(inputs, training=False)
        # Updates the metrics tracking the loss
        rec_loss = output['rec_loss']
        loss_value = rec_loss + output['kl_nodes'] + output['kl_decisions']
        # Update the metrics.
        metric = self.metric(y, output['p_c_z'])

        self.loss_value.update_state(loss_value)
        self.rec_loss.update_state(rec_loss)
        self.kl_decisions.update_state(output['kl_decisions'])
        self.kl_nodes.update_state(output['kl_nodes'])
        self.nmi.update_state(metric)

        return {"loss_value": self.loss_value.result(), "rec_loss": self.rec_loss.result(),
                "kl_decisions": self.kl_decisions.result(), "kl_nodes": self.kl_nodes.result(),
                "nmi": self.nmi.result(), 'alpha': self.alpha}
