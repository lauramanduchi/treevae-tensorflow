"""
TreeVAE model.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from utils.model_utils import compute_posterior
from models.networks import (get_encoder, get_decoder, MLP, Dense)
from utils.training_utils import calc_aug_loss

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


class LadderVAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LadderVAE, self).__init__(name="LadderVAE")
        self.kwargs = kwargs

        # saving important variables to initialize the tree
        self.encoded_sizes = self.kwargs['latent_dim']
        self.hidden_layers = self.kwargs['mlp_layers']
        # check that the number of layers for bottom up is equal to top down
        if len(self.encoded_sizes) != len(self.hidden_layers):
            raise ValueError('Model is mispecified!!')
        self.depth = self.kwargs['initial_depth']
        self.inp_shape = self.kwargs['inp_shape']
        self.activation = self.kwargs['activation']
        self.augment = self.kwargs['augment']
        self.augmentation_method = self.kwargs['augmentation_method']
        self.aug_decisions_weight = self.kwargs['aug_decisions_weight']

        # bottom up: the inference chain that from x computes the d units until the root
        if self.activation == "mse":
            size = int(tf.sqrt(self.inp_shape / 3))
            encoder = get_encoder(architecture=self.kwargs['encoder'], encoded_size=self.encoded_sizes[0],
                                hidden_layer=self.hidden_layers[0], size=size)
        else:
            encoder = get_encoder(architecture=self.kwargs['encoder'], encoded_size=self.encoded_sizes[0],
                                hidden_layer=self.hidden_layers[0])   
            
        self.bottom_up = [encoder]
        for i in range(1, len(self.hidden_layers)):
            self.bottom_up.append(MLP(self.encoded_sizes[i], self.hidden_layers[i]))

        # MLP's if we use contrastive loss on bottom-up embeddings
        if len([i for i in self.augmentation_method if i in ['instancewise_first', 'instancewise_full']])>0:
            self.contrastive_mlp = []
            for i in range(0, len(self.hidden_layers)):
                self.contrastive_mlp.append(MLP(encoded_size=64, hidden_unit=512))

        # select the top down generative networks
        encoded_size_gen = self.encoded_sizes[-(self.depth+1):] # e.g. encoded_sizes 32,16,8, depth 1
        encoded_size_gen = encoded_size_gen[::-1] # encoded_size_gen = 16,8 => 8,16
        encoded_size_gen = encoded_size_gen[1:] # encoded_size_gen = 16 (root does not have a transformation)
        layers_gen = self.hidden_layers[-(self.depth+1):] # e.g. encoded_sizes 256,128,64, depth 1
        layers_gen = layers_gen[::-1] # encoded_size_gen = 128,64 => 64,128
        layers_gen = layers_gen[:-1] # 64 as the leaves have decoder

        # add root transformation and dense layer, the dense layer is layer that connects the bottom-up with the nodes
        self.transformations = [None]
        self.denses = [Dense(self.encoded_sizes[-1])] # the dense layer has latent dim = the dim of the node
        for i in range(self.depth):
                self.transformations.append(MLP(encoded_size_gen[i], layers_gen[i]))
                self.denses.append(Dense(encoded_size_gen[i]))

        self.decoder = get_decoder(architecture=self.kwargs['encoder'], hidden_layer=self.hidden_layers[0],
                              input_shape=self.inp_shape, activation=self.activation)


    def call(self, inputs, training=True):
        epsilon = tf.keras.backend.epsilon()
        x = inputs
        x = tf.keras.layers.Flatten()(x)

        # compute deterministic bottom up
        d = x
        encoders = []
        emb_contr = []

        for i in range(0, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d, training)

            # Pass through contrastive MLP's
            if 'instancewise_full' in self.augmentation_method:
                _, emb_c, _ = self.contrastive_mlp[i](d)
                emb_contr.append(emb_c)
            elif 'instancewise_first' in self.augmentation_method:
                if i==0:
                    _, emb_c, _ = self.contrastive_mlp[i](d)
                    emb_contr.append(emb_c)


            # store only the layers that are used for the top down
            if i >= len(self.hidden_layers)-(self.depth+1):
                encoders.append(d)

        # initializate KL losses
        kl_nodes_tot = tf.zeros(1)
        aug_decisions_loss = tf.zeros(1)
        depth_level = 0
        z_parent_sample = None
        for i in range(self.depth+1):

            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[self.depth - depth_level]
            z_mu_q_hat, z_sigma_q_hat = self.denses[depth_level](d)

            if depth_level == 0:  
                # here we are in the root
                # standard gaussian
                z_mu_p, z_sigma_p = tf.zeros(tf.shape(z_mu_q_hat)), tf.ones(tf.shape(z_sigma_q_hat))
                z_p = tfd.MultivariateNormalDiag(loc=z_mu_p, scale_diag=tf.math.sqrt(z_sigma_p + epsilon))
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = self.transformations[depth_level](z_parent_sample, training)
                z_p = tfd.MultivariateNormalDiag(loc=z_mu_p, scale_diag=tf.math.sqrt(z_sigma_p + epsilon))
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)


            # compute sample z using mu_q and sigma_q
            z = tfd.MultivariateNormalDiag(loc=z_mu_q, scale_diag=tf.math.sqrt(z_sigma_q + epsilon))
            # trick used to save the sample from distr in tensorflow but basically is same as above
            mu_var_q = tf.concat([z_mu_q, tf.math.sqrt(z_sigma_q + epsilon)], axis=1)
            l = z_mu_q.shape[1]
            z_q_dist = tfp.layers.DistributionLambda(
                lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :l], scale_diag=theta[:, l:]))
            z_sample = z_q_dist(mu_var_q)


            z_parent_sample = z_sample

            # compute KL node
            kl_node = tf.clip_by_value(tfd.kl_divergence(z, z_p), clip_value_min=-1, clip_value_max=10E10)

            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            if depth_level == self.depth:
                reconstruction = self.decoder(z_sample)
            depth_level += 1

        kl_nodes_loss = tf.math.reduce_mean(tf.clip_by_value(kl_nodes_tot, clip_value_min=-10, clip_value_max=10E10))

        rec_losses = self.loss(x, reconstruction)
        
        if self.augment and 'simple' not in self.augmentation_method and training is True:
            aug_decisions_loss += calc_aug_loss(prob_parent=tf.ones(len(x)), prob_router=tf.zeros(len(x)), augmentation_methods=self.augmentation_method, emb_contr=emb_contr)

        return {
            'rec': reconstruction,
            'rec_loss': tf.math.reduce_mean(rec_losses),
            'kl_root': tf.math.reduce_mean(kl_root),
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'z_leaves': z_sample,
            'elbo_samples': kl_nodes_tot + kl_root + rec_losses
        }

    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss
        self.alpha = tf.Variable(0.0, trainable=False)
        # Metrics
        self.loss_value = tfk.metrics.Mean(name="loss_value")
        self.rec_loss = tfk.metrics.Mean(name="rec_loss")
        self.kl_root = tfk.metrics.Mean(name="kl_root")
        self.kl_nodes = tfk.metrics.Mean(name="kl_nodes")
        self.aug_decisions = tfk.metrics.Mean(name='aug_decisions')

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            output = self(x, training=True)
            rec_loss = output['rec_loss']
            loss_value = rec_loss + self.alpha * (output['kl_root'] + output['kl_nodes'] ) + output['aug_decisions']

        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_value.update_state(loss_value)
        self.rec_loss.update_state(rec_loss)
        self.kl_root.update_state(output['kl_root'])
        self.kl_nodes.update_state(output['kl_nodes'])
        self.aug_decisions.update_state(output['aug_decisions'])

        return {"loss_value": self.loss_value.result(), "rec_loss": self.rec_loss.result(), "kl_root": self.kl_root.result(),
                "kl_nodes": self.kl_nodes.result(), "aug_decisions": self.aug_decisions.result(), 'alpha': self.alpha}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_value, self.rec_loss, self.kl_root, self.kl_nodes, self.aug_decisions]


    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        output = self(x, training=False)
        # Updates the metrics tracking the loss
        rec_loss = output['rec_loss']
        loss_value = rec_loss + output['kl_root'] + output['kl_nodes']

        self.loss_value.update_state(loss_value)
        self.rec_loss.update_state(rec_loss)
        self.kl_root.update_state(output['kl_root'])
        self.kl_nodes.update_state(output['kl_nodes'])

        return {"loss_value": self.loss_value.result(), "rec_loss": self.rec_loss.result(), "kl_root": self.kl_root.result(),
                 "kl_nodes": self.kl_nodes.result(), 'alpha': self.alpha}