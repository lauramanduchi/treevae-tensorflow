"""
Utility functions for training.
"""
import tensorflow as tf
import math
import numpy as np
import wandb

class AnnealKLCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.01, start=0.):
        self.decay = decay
        self.start = start

    def on_epoch_end(self, epoch, logs=None):
        value = self.start + epoch * self.decay
        self.model.alpha.assign(min(1, value))

class Decay():
    def __init__(self, lr=0.001, drop=0.1, epochs_drop=50):
        self.lr = lr
        self.drop = drop
        self.epochs_drop = epochs_drop

    def learning_rate_scheduler(self, epoch):
        initial_lrate = self.lr
        drop = self.drop
        epochs_drop = self.epochs_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
    

def calc_aug_loss(prob_parent, prob_router, augmentation_methods, emb_contr=[]):
    aug_decisions_loss = tf.zeros(1)
    prob_parent = tf.stop_gradient(prob_parent)

    num_losses = len(augmentation_methods)
    if emb_contr == [] and 'instancewise_first' in augmentation_methods:
        num_losses = num_losses - 1
    if emb_contr == [] and 'instancewise_full' in augmentation_methods:
        num_losses = num_losses - 1
    if num_losses<=0:
        # If only instancewise losses and we're in smalltree
        return aug_decisions_loss

    # Get router probabilities of X' and X''
    p1, p2 = prob_router[:len(prob_router)//2], prob_router[len(prob_router)//2:]
    # Perform invariance regularization
    for aug_method in augmentation_methods:
        if aug_method == 'InfoNCE':
            p1_normed = tf.math.l2_normalize(tf.stack([p1,1-p1],1), axis=1)
            p2_normed = tf.math.l2_normalize(tf.stack([p2,1-p2],1), axis=1)
            pair_sim = tf.exp(tf.reduce_sum(p1_normed * p2_normed,axis = 1))
            p_normed = tf.concat([p1_normed,p2_normed],axis=0)
            matrix_sim = tf.exp(p_normed @ tf.transpose(p_normed))
            norm_factor = tf.reduce_sum(matrix_sim, axis=1) - tf.linalg.diag_part(matrix_sim)
            pair_sim = tf.tile(pair_sim,[2]) # storing sim for X' and X''
            info_nce_sample = -tf.math.log(pair_sim / norm_factor)
            info_nce = tf.math.reduce_sum(prob_parent * info_nce_sample) / tf.math.reduce_sum(prob_parent)
            aug_decisions_loss += info_nce

        elif aug_method in ['instancewise_first','instancewise_full']:
            looplen = len(emb_contr) if aug_method == 'instancewise_full' else min(len(emb_contr),1)
            for i in range(looplen):
                temp_instance = 0.5
                emb = emb_contr[i]
                emb1, emb2 = emb[:len(emb)//2], emb[len(emb)//2:]
                emb1_normed = tf.math.l2_normalize(emb1, axis=1)
                emb2_normed = tf.math.l2_normalize(emb2, axis=1)
                pair_sim = tf.exp(tf.reduce_sum(emb1_normed * emb2_normed,axis = 1) / temp_instance)
                emb_normed = tf.concat([emb1_normed,emb2_normed],axis=0)
                matrix_sim = tf.exp(emb_normed @ tf.transpose(emb_normed) / temp_instance)
                norm_factor = tf.reduce_sum(matrix_sim, axis=1) - tf.linalg.diag_part(matrix_sim)
                pair_sim = tf.tile(pair_sim,[2]) # storing sim for X' and X''
                info_nce_sample = -tf.math.log(pair_sim / norm_factor)
                info_nce = tf.math.reduce_mean(info_nce_sample)
                info_nce = info_nce / looplen
                aug_decisions_loss += info_nce

        else: raise NotImplementedError

    # Also take into account that for smalltree, instancewise losses are 0      
    aug_decisions_loss = aug_decisions_loss / num_losses
    
    return aug_decisions_loss




def get_data_small_tree(x, y, node_leaves, n_effective_leaves):
	z = node_leaves['z_sample']
	prob = node_leaves['prob']
	ind = np.where(prob >= min(1/n_effective_leaves,0.5)) # To circumvent problems with n_effective_leaves==1
	x_small, y_small = x[ind], y[ind]
	z_small, prob_small, = z[ind], prob[ind]
	return x_small, z_small, prob_small, y_small


def compute_leaves(tree):
	list_nodes = [{'node': tree, 'depth': 0}]
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
			# We are in an internal node with pruned leaves and thus only have one child
			node_left, node_right = node.left, node.right
			child = node_left if node_left is not None else node_right
			list_nodes.append(
				{'node': child, 'depth': depth_level + 1})
		else:
			nodes_leaves.append(current_node)
	return nodes_leaves


def check_conditions_growing(y, y_small):
	# check whether the selected node contains more than one digit
	digits, counts = np.unique(y, return_counts=True)
	digits_small, counts_small = np.unique(y_small, return_counts=True)
	c = 0
	for i in digits:
		if i in digits_small:
			ind = np.argwhere(digits_small == i)[0][0]
			m = counts_small[ind]
			ind = np.argwhere(digits == i)[0][0]
			n = counts[ind]
			if m / n > 0.5:
				c += 1
				if c > 1:
					return True
	return False


def compute_growing_leaf(x_train, y_train, model, output_train, max_depth, batch_size):
	leaves = compute_leaves(model.tree)
	split_oracle = []
	n_samples = []
	node_leaves_train = output_train['node_leaves']
	weights = [node_leaves_train[i]['prob'] for i in range(len(node_leaves_train))]
	weights_summed = [weights[i].sum() for i in range(len(weights))]
	n_effective_leaves = len(np.where(weights_summed/np.sum(weights_summed) >= 0.01)[0])

	# Calculating ground-truth nodes-to-split for logging and model development
	for i in range(len(node_leaves_train)):
		depth, node = leaves[i]['depth'], leaves[i]['node']
		if not node.expand:
			continue
		_,_,_, y_train_small = \
			get_data_small_tree(x_train, y_train, node_leaves_train[i], n_effective_leaves)

		cg = check_conditions_growing(y_train, y_train_small)
		print(np.unique(y_train_small, return_counts=True))
		print(cg)

		split_oracle.append(cg)
		n_samples.append(len(y_train_small))
	
	split_values = n_samples
	# Highest number of samples indicates splitting
	ind_leaves = np.argsort(np.array(split_values))
	ind_leaves = ind_leaves[::-1]


	print(ind_leaves)
	for i in ind_leaves:
		if n_samples[i] < batch_size:
			wandb.log({'Skipped Split': 1})
			print("We don't split leaves with fewer samples than batch size")
			continue
		elif leaves[i]['depth'] == max_depth or not leaves[i]['node'].expand:
			leaves[i]['node'].expand = False
			print('\n!!ATTENTION!! architecture is not deep enough\n')
			break
		else:
			ind_leaf = i
			leaf = leaves[ind_leaf]
			wandb.log({'Valid Split': int(split_oracle[ind_leaf])})
			return ind_leaf, leaf

	return None, None


def compute_pruning_leaf(model, output_train):    
	leaves = compute_leaves(model.tree)
	node_leaves_train = output_train['node_leaves']
	n_leaves = len(output_train['node_leaves'])
	weights = [node_leaves_train[i]['prob'] for i in range(n_leaves)]

	# Assign each sample to a leaf by argmax(weights)
	max_indeces = np.array([np.argmax(col) for col in zip(*weights)])
	n_samples = []
	for i in range(n_leaves):
		n_samples.append(sum(max_indeces==i))

    # Prune leaves with less than 1% of all samples
	ind_leaf = np.argmin(n_samples)
	if n_samples[ind_leaf] < 0.01 * sum(n_samples):
		leaf = leaves[ind_leaf]
		return ind_leaf, leaf
	else:
		return None, None


def get_optimizer(configs):
	try:
		optimizer = tf.keras.optimizers.Adam(learning_rate=configs['training']['lr'],
											 decay=configs['training']['decay'])
	except:
		optimizer = tf.keras.optimizers.Adam(learning_rate=configs['training']['lr'],
											 weight_decay=configs['training']['decay'])
	return optimizer