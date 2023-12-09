import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gc

from utils.utils import accuracy_metric
from utils.training_utils import AnnealKLCallback, Decay, get_data_small_tree, check_conditions_growing, compute_growing_leaf, compute_pruning_leaf, get_optimizer
from utils.data_utils import get_gen, DataGenSmallTree
from utils.model_utils import return_list_tree, construct_data_tree
from models.model import TreeVAE
from models.model_smalltree import SmallTreeVAE

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


def run_tree(x_train, x_test, y_train, y_test, configs, loss):

	gen_train = get_gen(x_train, y_train, configs, configs['training']['batch_size'])
	gen_test = get_gen(x_test, y_test, configs, configs['training']['batch_size'], validation=True)
	_ = gc.collect()

	# Define model & optimizer
	model = TreeVAE(**configs['training'])
	optimizer = get_optimizer(configs)

	# Training the initial split
	model.compile(optimizer, loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
	lrscheduler = Decay(lr=configs['training']['lr'], drop=0.1, epochs_drop=100).learning_rate_scheduler
	cp_callback_scheduler = [WandbCallback(), AnnealKLCallback(decay=configs['training']['decay_kl'], start=configs['training']['kl_start']),
							 tf.keras.callbacks.LearningRateScheduler(lrscheduler)]
	model.fit(gen_train, validation_data=gen_test, callbacks=cp_callback_scheduler, epochs=configs['training']['num_epochs'], verbose=2)
	_ = gc.collect()

	# Start the growing loop of the tree
	# Compute metrics and set node.expand False for the nodes that should not grow
	# This loop goes layer-wise
	grow = configs['training']['grow']
	initial_depth = configs['training']['initial_depth']
	max_depth = len(configs['training']['mlp_layers']) - 1
	if initial_depth >= max_depth:
		grow = False
	growing_iterations = 0
	while grow and growing_iterations<150:

		# full model finetuning during growing every 3 splits
		if configs['training']['grow_fulltrain']:
			if growing_iterations != 0 and growing_iterations % 3 == 0:
				lrscheduler = Decay(lr=configs['training']['lr'], drop=0.1, epochs_drop=40).learning_rate_scheduler
				print('\nTree finetuning\n')
				model.compile(get_optimizer(configs), loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
				model.fit(gen_train, validation_data=gen_test, callbacks=[WandbCallback(), tf.keras.callbacks.LearningRateScheduler(lrscheduler), 
								  AnnealKLCallback(decay=configs['training']['decay_kl'], start=configs['training']['kl_start'])], epochs=80, verbose=2)
				_ = gc.collect()


		model.compile(optimizer, loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
		output_train = model.predict(x_train, batch_size=configs['training']['batch_size'])
		output_test = model.predict(x_test, batch_size=configs['training']['batch_size'])
		node_leaves_train = output_train['node_leaves']
		node_leaves_test = output_test['node_leaves']

		# count effective number of leaves 
		weights = [node_leaves_train[i]['prob'] for i in range(len(node_leaves_train))]
		weights_summed = [weights[i].sum() for i in range(len(weights))]
		weights = weights_summed/np.sum(weights_summed)
		length = len(np.where(weights > 0.01)[0])
		print(length)

		# grow until reaching required n_effective_leaves 
		growing = length < configs['training']['num_clusters_tree']

		if not growing:
			print('\nReached maximum number of leaves\n')
			break

		# compute which leaf to split
		ind_leaf, leaf = compute_growing_leaf(x_train, y_train, model, output_train, max_depth, configs['training']['batch_size'])

		if ind_leaf == None:
			print('\nReached maximum architecture\n')
			break

		print('\nGrowing tree: Node %d, depth %d\n' % (ind_leaf, leaf['depth']))


		depth, node = leaf['depth'], leaf['node']
		if not node.expand:
			continue

		# get subset of data that has high prob. of falling in subtree
		x_train_small, z_train_small, prob_train_small, y_train_small = get_data_small_tree(x_train, y_train, node_leaves_train[ind_leaf], length)
		x_test_small, z_test_small, prob_test_small, y_test_small = get_data_small_tree(x_test, y_test, node_leaves_test[ind_leaf], length)

		condition = check_conditions_growing(y_train, y_train_small)
		print(condition)
		print(np.unique(y_train_small, return_counts=True))


		# initialize the smalltree dataloader with the correct augmentation scheme
		if configs['training']['augment']:
			if configs['data']['data_name'] == 'omniglot':
				gen_train_small = DataGenSmallTree(x_train_small, y_train_small, model=model,ind_leaf=ind_leaf,augment=True, augmentation_method='omniglot',batch_size=configs['training']['batch_size'])
			elif len([i for i in configs['training']['augmentation_method'] if i in ['InfoNCE']])>0:
				gen_train_small = DataGenSmallTree(x_train_small, y_train_small, model=model,ind_leaf=ind_leaf, 
												augment=configs['training']['augment'], augmentation_method=configs['training']['augmentation_method'],
												batch_size=configs['training']['batch_size'], dataset=configs['data']['data_name'])
			else:
				# If simple OR only instancewise methods, we train subtrees with augmentations w/o contrastive losses.
				# w/o because d's are fixed. Still using Xaug instead of X because model will see Xaug again during finetuning
				gen_train_small = DataGenSmallTree(x_train_small, y_train_small, model=model,ind_leaf=ind_leaf, 
												augment=configs['training']['augment'], augmentation_method=['simple'],
												batch_size=configs['training']['batch_size'], dataset=configs['data']['data_name'])				
		else:
			gen_train_small = DataGenSmallTree(x_train_small, y_train_small, z=z_train_small,prob=prob_train_small,
											batch_size=configs['training']['batch_size'])


		gen_test_small = DataGenSmallTree(x_test_small, y_test_small, z=z_test_small,prob=prob_test_small,
										batch_size=configs['training']['batch_size'])
		
		# initialize the smalltree
		new_depth = depth + 1
		small_model = SmallTreeVAE(new_depth, model.bottom_up, **configs['training'])

		# train the smalltree
		small_model.compile(get_optimizer(configs), loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
		lrscheduler = Decay(lr=configs['training']['lr'], drop=0.1, epochs_drop=100).learning_rate_scheduler
		cp_callback_scheduler = [WandbCallback(), tf.keras.callbacks.LearningRateScheduler(lrscheduler),
									AnnealKLCallback(decay=configs['training']['decay_kl'], start=configs['training']['kl_start'])]
		small_model.fit(gen_train_small, validation_data=gen_test_small, callbacks=cp_callback_scheduler,
						epochs=150, verbose=2)

		# attach smalltree to full tree by assigning decisions and adding new children nodes to full tree
		model.attach_smalltree(node, small_model)

		model.compile(optimizer, loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
		metrics = model.evaluate(x_test, y_test, batch_size=configs['training']['batch_size'], return_dict=True)
		print(metrics)
		model_depth = model.compute_depth()

		# Check if reached the max number of effective leaves 
		if length+1 == configs['training']['num_clusters_tree']:
			output_train = model.predict(x_train, batch_size=configs['training']['batch_size'])
			node_leaves_train = output_train['node_leaves']
			weights = [node_leaves_train[i]['prob'] for i in range(len(node_leaves_train))]
			weights_summed = [weights[i].sum() for i in range(len(weights))]
			weights = weights_summed/np.sum(weights_summed)
			length_new = len(np.where(weights > 0.01)[0])
			growing = length_new < configs['training']['num_clusters_tree'] 
			if not growing:
				print('\nReached maximum number of leaves\n')
				break	

		growing_iterations += 1

	# check whether we prune and log pre-pruning dendrogram
	prune = configs['training']['prune']
	if prune:
		output_test = model.predict(x_test, batch_size=configs['training']['batch_size'])
		if len(output_test['node_leaves'])<2:
			prune = False
		else:
			print('\nStarting pruning!\n')
			p_c_z = output_test['p_c_z']
			yy = np.squeeze(np.argmax(p_c_z, axis=-1))
			data_tree = construct_data_tree(model, y_predicted=yy, y_true=y_test, n_leaves=len(output_test['node_leaves']),
											data_name=configs['data']['data_name'])

			table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree)
			fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
			dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
			wandb.log({"dendogram_pre_pruned": dendro})

	# prune the tree
	while prune:
		output_train = model.predict(x_train, batch_size=configs['training']['batch_size'])
		# check pruning conditions
		ind_leaf, leaf = compute_pruning_leaf(model, output_train)
		if ind_leaf == None:
			print('\nPruning finished!\n')
			break
		else:
			# prune leaves and internal nodes without children
			current_node = leaf['node']
			while all(child is None for child in [current_node.left, current_node.right]):
				if current_node.parent is not None:
					parent = current_node.parent
				# root does not get pruned
				else:
					break
				parent.prune_child(current_node)
				current_node = parent


			# reinitialize model
			transformations, routers, denses, decoders, routers_q = return_list_tree(model.tree)
			model.decisions_q = routers_q
			model.transformations = transformations
			model.decisions = routers
			model.denses = denses
			model.decoders = decoders
			model.depth = model.compute_depth()

			model.compile(optimizer, loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
			_ = gc.collect()


	print('\n*****************model depth %d******************\n' % (model_depth))
	print('\n*****************model finetuning******************\n')

	# finetune the full tree
	model.compile(tf.keras.optimizers.Adam(learning_rate=configs['training']['lr']), loss=loss, metric=accuracy_metric, alpha=configs['training']['kl_start'])
	lrscheduler = Decay(lr=configs['training']['lr'], drop=0.1, epochs_drop=100).learning_rate_scheduler
	model.fit(gen_train, validation_data=gen_test, callbacks=[WandbCallback(), tf.keras.callbacks.LearningRateScheduler(lrscheduler),
									AnnealKLCallback(decay=0.01, start=configs['training']['kl_start'])], epochs=configs['training']['num_epochs_finetuning'],verbose=2)
	_ = gc.collect()

	return model


