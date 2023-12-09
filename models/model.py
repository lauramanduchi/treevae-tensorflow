"""
TreeVAE model.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from utils.model_utils import construct_tree, compute_posterior
from models.networks import (get_encoder, get_decoder, MLP, Router, Dense)
from utils.model_utils import return_list_tree
from utils.training_utils import calc_aug_loss

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


class TreeVAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(TreeVAE, self).__init__(name="TreeVAE")
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
        self.return_x = False 

        # bottom up: the inference chain that from x computes the d units till the root
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

        # MLP's if we use contrastive loss on d's
        if len([i for i in self.augmentation_method if i in ['instancewise_first', 'instancewise_full']])>0:
            self.contrastive_mlp = []
            for i in range(0, len(self.hidden_layers)):
                self.contrastive_mlp.append(MLP(encoded_size=64, hidden_unit=512))

        # top down: the generative model that from x computes the prior prob of all nodes from root till leaves
        # it has a tree structure which is constructed by passing a list of transformations and routers from root to
        # leaves visiting nodes layer-wise from left to right
        # N.B. root has None as transformation and leaves have None as routers
        # the encoded sizes and layers are reversed from bottom up
        # e.g. for bottom up [MLP(256, 32), MLP(128, 16), MLP(64, 8)] the list of top-down transformations are
        # [None, MLP(16, 64), MLP(16, 64), MLP(32, 128), MLP(32, 128), MLP(32, 128), MLP(32, 128)]

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
            for j in range(2 ** (i + 1)):
                self.transformations.append(MLP(encoded_size_gen[i], layers_gen[i]))
                self.denses.append(Dense(encoded_size_gen[i]))

        self.decisions = []
        for i in range(self.depth):
            for j in range(2 ** i):
                self.decisions.append(Router(hidden_units=layers_gen[i]))

        # decoders = [None, None, None, Dec, Dec, Dec, Dec]
        self.decoders = [None for i in range(self.depth) for j in range(2 ** i)]
        # the leaves do not have decisions but have decoders
        for i in range(2 ** (self.depth)):
            self.decisions.append(None)
            self.decoders.append(get_decoder(architecture=self.kwargs['encoder'], hidden_layer=self.hidden_layers[0],
                              input_shape=self.inp_shape, activation=self.activation))

        # bottom-up decisions
        self.decisions_q = []
        for i in range(self.depth):
            for j in range(2 ** i):
                self.decisions_q.append(Router(hidden_units=layers_gen[i]))
        for i in range(2 ** (self.depth)):
            self.decisions_q.append(None)

        # construct the tree
        self.tree = construct_tree(transformations=self.transformations, routers=self.decisions,
                                        routers_q=self.decisions_q, denses=self.denses, decoders=self.decoders)

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

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': tf.ones(len(x)), 'z_parent_sample': None}]
        # initializate KL losses
        kl_nodes_tot = tf.zeros(1)
        kl_decisions_tot = tf.zeros(1)
        aug_decisions_loss = tf.zeros(1)
        leaves_prob = []
        reconstructions = []
        node_leaves = []
        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[self.depth - depth_level]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:  
                # here we are in the root
                # standard gaussian
                z_mu_p, z_sigma_p = tf.zeros(tf.shape(z_mu_q_hat)), tf.ones(tf.shape(z_sigma_q_hat))
                z_p = tfd.MultivariateNormalDiag(loc=z_mu_p, scale_diag=tf.math.sqrt(z_sigma_p + epsilon))
                # sampled z is the top layer of deterministic bottom-up
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample, training)
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

            # compute KL node
            kl_node = tf.clip_by_value(prob * tfd.kl_divergence(z, z_p),clip_value_min=-1,clip_value_max=1000)
        
            if depth_level == 0:
                kl_root = kl_node
            else:
                kl_nodes_tot += kl_node

            if node.router is not None:
                # we are in the internal nodes (not leaves)
                prob_child_left = tf.squeeze(node.router(z_sample, training))
                router_q = node.routers_q
                prob_child_left_q = tf.squeeze(router_q(d, training))

                kl_decisions = prob_child_left_q * tf.math.log(epsilon + prob_child_left_q / (prob_child_left+epsilon)) + \
                                (1 - prob_child_left_q) * tf.math.log(epsilon + (1 - prob_child_left_q) / (1 - prob_child_left+epsilon))
                
                if self.augment and 'simple' not in self.augmentation_method and training is True:
                    if depth_level == 0:
                        # Only do contrastive loss on representations once
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=emb_contr)
                    else:
                        aug_decisions_loss += calc_aug_loss(prob_parent=prob, prob_router=prob_child_left_q, augmentation_methods=self.augmentation_method, emb_contr=[])


                kl_decisions = prob * kl_decisions
                kl_decisions_tot += kl_decisions

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                   'z_parent_sample': z_sample})
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_dist': (z_mu_q, z_sigma_q),
                                    'z_sample': z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        kl_nodes_loss = tf.clip_by_value(tf.math.reduce_mean(kl_nodes_tot), clip_value_min=-10, clip_value_max=10E10)
        kl_decisions_loss = tf.math.reduce_mean(kl_decisions_tot)
        kl_root_loss = tf.math.reduce_mean(kl_root)

        # p_c_z is the probability of reaching a leaf and should be of shape [batch_size, num_clusters]
        p_c_z = tf.concat([tf.expand_dims(leaves_prob[i], -1) for i in range(len(leaves_prob))], axis=-1)
        p_c_z = tf.cast(p_c_z, tf.float64)
        
        rec_losses = self.loss(x, reconstructions, leaves_prob)
        rec_loss = tf.math.reduce_mean(rec_losses, axis=0)


        return_dict = {
            'rec_loss': rec_loss,
            'weights': leaves_prob,
            'kl_root': kl_root_loss,
            'kl_decisions': kl_decisions_loss,
            'kl_nodes': kl_nodes_loss,
            'aug_decisions': self.aug_decisions_weight * aug_decisions_loss,
            'p_c_z': p_c_z,
            'node_leaves': node_leaves,
            'elbo_samples': kl_nodes_tot+kl_decisions_tot+kl_root+rec_losses
        }

        if self.return_x:
            return_dict['input'] = x

        return return_dict

    def compile(self, optimizer, loss, metric, alpha=0.):
        super().compile(optimizer)
        self.loss = loss
        self.metric = metric
        self.alpha = tf.Variable(alpha, trainable=False)
        # Metrics
        self.loss_value = tfk.metrics.Mean(name="loss_value")
        self.rec_loss = tfk.metrics.Mean(name="rec_loss")
        self.kl_root = tfk.metrics.Mean(name="kl_root")
        self.kl_decisions = tfk.metrics.Mean(name="kl_decisions")
        self.kl_nodes = tfk.metrics.Mean(name="kl_nodes")
        self.nmi = tfk.metrics.Mean(name="nmi")
        self.aug_decisions = tfk.metrics.Mean(name='aug_decisions')
        self.perc_samples = tfk.metrics.Mean(name="perc_left")

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            output = self(x, training=True)
            rec_loss = output['rec_loss']
            loss_value = rec_loss + self.alpha * (output['kl_root'] + output['kl_decisions'] + output['kl_nodes']) + output['aug_decisions']
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        metric = self.metric(y, output['p_c_z'])


        # Monitor splits
        if self.depth == 1:
            self.perc_samples.update_state(1 - tf.reduce_mean(tf.cast(tf.argmax(output['p_c_z'], axis=-1), "float")))
        else:
            self.perc_samples.update_state(0)
        self.loss_value.update_state(loss_value)
        self.rec_loss.update_state(rec_loss)
        self.kl_root.update_state(output['kl_root'])
        self.kl_decisions.update_state(output['kl_decisions'])
        self.kl_nodes.update_state(output['kl_nodes'])
        self.nmi.update_state(metric)
        self.aug_decisions.update_state(output['aug_decisions'])

        return {"loss_value": self.loss_value.result(), "rec_loss": self.rec_loss.result(), "kl_root": self.kl_root.result(),
                "kl_decisions": self.kl_decisions.result(), "aug_decisions": self.aug_decisions.result(), "kl_nodes": self.kl_nodes.result(),
                "nmi": self.nmi.result(), 'perc_left': self.perc_samples.result(), 'alpha': self.alpha}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_value, self.rec_loss,self.kl_root,self.kl_decisions,self.kl_nodes,self.nmi,self.aug_decisions,self.perc_samples] # ,self.elbo_bpd,self.rec_bpd


    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        output = self(x, training=False)
        # Updates the metrics tracking the loss
        rec_loss = output['rec_loss']
        loss_value = rec_loss + output['kl_root'] + output['kl_decisions'] + output['kl_nodes']

        # Update the metrics.
        metric = self.metric(y, output['p_c_z'])

        self.loss_value.update_state(loss_value)
        self.rec_loss.update_state(rec_loss)
        self.kl_root.update_state(output['kl_root'])
        self.kl_decisions.update_state(output['kl_decisions'])
        self.kl_nodes.update_state(output['kl_nodes'])
        self.nmi.update_state(metric)
        return_dict = {"loss_value": self.loss_value.result(), "rec_loss": self.rec_loss.result(), "kl_root": self.kl_root.result(),
                "kl_decisions": self.kl_decisions.result(), "kl_nodes": self.kl_nodes.result(), "nmi": self.nmi.result(), 'alpha': self.alpha}

        return return_dict

    def compute_leaves(self):
        # returns leaves of the tree
        list_nodes = [{'node': self.tree, 'depth': 0}]
        nodes_leaves = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level = current_node['node'], current_node['depth']
            if node.router is not None:
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1})
            elif node.router is None and node.decoder is None:
                # we are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append({'node': child, 'depth': depth_level + 1})                
            else:
                nodes_leaves.append(current_node)
        return nodes_leaves


    def compute_depth(self):
        # computes depth of the tree
        nodes_leaves = self.compute_leaves()
        d = []
        for i in range(len(nodes_leaves)):
            d.append(nodes_leaves[i]['depth'])
        return max(d)

    def attach_smalltree(self, node, small_model):
        # attaching a (trained) smalltree to the full tree
        assert node.left is None and node.right is None
        node.router = small_model.decision
        node.routers_q = small_model.decision_q
        node.decoder = None
        for j in range(2):
            dense = small_model.denses[j]
            transformation = small_model.transformations[j]
            decoder = small_model.decoders[j]
            node.insert(transformation, None, None, dense, decoder)

        transformations, routers, denses, decoders, routers_q = return_list_tree(self.tree)
        self.decisions_q = routers_q

        self.transformations = transformations
        self.decisions = routers
        self.denses = denses
        self.decoders = decoders
        self.depth = self.compute_depth()
        return


    def compute_reconstruction(self, inputs, training=False):
        epsilon = tf.keras.backend.epsilon()
        x = inputs
        # compute deterministic bottom up
        d = x
        encoders = []

        for i in range(0, len(self.hidden_layers)):
            d, _, _ = self.bottom_up[i](d, training)
            # store only the layers that are used for the top down
            if i >= len(self.hidden_layers) - (self.depth + 1):
                encoders.append(d)

        # create a list of nodes of the tree that need to be processed
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': tf.ones(len(x)), 'z_parent_sample': None}]
        # initializate KL losses

        leaves_prob = []
        reconstructions = []
        node_leaves = []
        while len(list_nodes) != 0:
            # store info regarding the current node
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']
            # access deterministic bottom up mu and sigma hat (computed above)
            d = encoders[self.depth - depth_level]
            z_mu_q_hat, z_sigma_q_hat = node.dense(d)

            if depth_level == 0:  # here we are in the root
                z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
            else:
                # the generative mu and sigma is the output of the top-down network given the sampled parent
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample, training)
                z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)


            # compute sample z using mu_q and sigma_q
            z = tfd.MultivariateNormalDiag(loc=z_mu_q, scale_diag=tf.math.sqrt(z_sigma_q + epsilon))
            # trick used to save the sample from distr in tensorflow but basically is same as above
            mu_var_q = tf.concat([z_mu_q, tf.math.sqrt(z_sigma_q + epsilon)], axis=1)
            l = z_mu_q.shape[1]
            z_q_dist = tfp.layers.DistributionLambda(
                lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :l], scale_diag=theta[:, l:]))
            z_sample = z_q_dist(mu_var_q)


            # if we are in the internal nodes (not leaves)
            if node.router is not None:
                prob_child_left_q = tf.squeeze(node.routers_q(d, training))

                # we are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                   'z_parent_sample': z_sample})
            elif node.decoder is not None:
                # if we are in a leaf we need to store the prob of reaching that leaf and compute reconstructions
                # as the nodes are explored left to right, these probabilities will be also ordered left to right
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))
                node_leaves.append({'prob': prob, 'z_dist': (z_mu_q, z_sigma_q),
                                    'z_sample': z_sample})

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

        return reconstructions, node_leaves

    def generate_images(self, n_samples):
        sizes = self.encoded_sizes
        list_nodes = [{'node': self.tree, 'depth': 0, 'prob': 1, 'z_parent_sample': None}]
        leaves_prob = []
        reconstructions = []
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
            z_parent_sample = current_node['z_parent_sample']

            if depth_level == 0:
                z_mu_p, z_sigma_p = tf.zeros([n_samples, sizes[-1]]), tf.ones([n_samples, sizes[-1]])
                z_p = tfd.MultivariateNormalDiag(loc=z_mu_p, scale_diag=tf.math.sqrt(z_sigma_p))
                z_sample = tf.squeeze(z_p.sample(1))
            else:
                _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample, training=False)
                z_p = tfd.MultivariateNormalDiag(loc=z_mu_p, scale_diag=tf.math.sqrt(z_sigma_p))
                z_sample = tf.squeeze(z_mu_p)

            if node.router is not None:
                prob_child_left = node.router(z_sample)
                prob_node_left, prob_node_right = prob * tf.squeeze(prob_child_left), prob * (
                        1 - tf.squeeze(prob_child_left))
                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                   'z_parent_sample': z_sample})

            elif node.decoder is not None:
                leaves_prob.append(prob)
                dec = node.decoder
                reconstructions.append(dec(z_sample))

            elif node.router is None and node.decoder is None:
                # We are in an internal node with pruned leaves and thus only have one child
                node_left, node_right = node.left, node.right
                child = node_left if node_left is not None else node_right
                list_nodes.append(
                    {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})


        p_c_z = tf.concat([tf.expand_dims(leaves_prob[i], -1) for i in range(len(leaves_prob))], axis=-1)
        p_c_z = tf.cast(p_c_z, tf.float64)
        return reconstructions, p_c_z