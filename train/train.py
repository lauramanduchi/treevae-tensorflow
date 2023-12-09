import time
from pathlib import Path
import wandb
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import uuid
import os
import gc
import scipy
import yaml


import utils.utils as utils

from utils.data_utils import get_data
from utils.utils import reset_random_seeds
from utils.training_utils import compute_leaves
from utils.model_utils import construct_data_tree
from train.train_tree import run_tree

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


def run_experiment(configs, loss):
	# Set paths
	project_dir = Path(__file__).absolute().parent
	timestr = time.strftime("%Y%m%d-%H%M%S")
	ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
	experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
	experiment_path.mkdir(parents=True)
	os.makedirs(os.path.join(project_dir, '../models/logs', ex_name))
	print(experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
	wandb.init(
		project="TreeVAE",
		entity="replace_by_your_wandb_entity",
		config=configs, 
		mode=configs['globals']['wandb_logging']
	)
	if configs['globals']['wandb_logging'] in ['online', 'disabled']:
		wandb.run.name = wandb.run.name.split("-")[-1] + "-"+ configs['run_name']
	elif configs['globals']['wandb_logging'] == 'offline':
		wandb.run.name = configs['run_name']
	else:
		raise ValueError('wandb needs to be set to online, offline or disabled.')

	# Reproducibility
	reset_random_seeds(configs['globals']['seed'])
	if configs['globals']['eager_mode'] and configs['globals']['wandb_logging']!='offline':
		tf.config.run_functions_eagerly(True)

	# Generate a new dataset each run
	x_train, x_test, y_train, y_test = get_data(configs)

	model = run_tree(x_train, x_test, y_train, y_test, configs, loss)

	# Save model
	if configs['globals']['save_model']:
		checkpoint_path = experiment_path
		print("\nSaving weights at ", experiment_path)
		model.save_weights(checkpoint_path)

	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)

	# Training set performance
	output = model.predict(x_train, batch_size=256)
	p_c_z = output['p_c_z']
	_ = gc.collect()
	if configs['globals']['save_model']:
		with open(experiment_path / 'c_train.npy', 'wb') as save_file:
			np.save(save_file, p_c_z)
	yy = np.squeeze(np.argmax(p_c_z, axis=-1))
	acc, idx = utils.cluster_acc(y_train, yy, return_index=True)
	swap = dict(zip(range(len(idx)), idx))
	y_wandb = np.array([swap[i] for i in yy], dtype=np.uint8)
	wandb.log({"Train_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
																	 y_true=y_train, preds=y_wandb,
																	 class_names=range(len(idx)))})
	nmi = normalized_mutual_info_score(y_train, yy)
	ari = adjusted_rand_score(y_train, yy)
	wandb.log({"Train Accuracy": acc, "Train Normalized Mutual Information": nmi, "Train Adjusted Rand Index": ari})
	file_results = "results_" + configs['data']['data_name'] + ".txt"

	f = open(file_results, "a+")
	f.write(
		"Epochs= %d, batch_size= %d, digits= %s, latent_dim= %s, K= %d, learning_rate= %f, decay= %f, encoder= %s, name= %s, "
		"seed= %d, config= %s.\n"
		% (configs['training']['num_epochs'], configs['training']['batch_size'],  np.unique(y_train), configs['training']['latent_dim'],
		   configs['training']['num_clusters_tree'], configs['training']['lr'], configs['training']['decay'],
		   configs['training']['encoder'], ex_name, configs['globals']['seed'], configs['globals']['config_name']))

	f.write("Train |   Accuracy: %.3f, NMI: %.3f, ARI: %.3f. \n" % (
		acc, nmi, ari))

	# Test set performance
	metrics = model.evaluate(x_test, y_test, batch_size=configs['training']['batch_size'], return_dict=True)
	metrics = {f'test_{k}': v for k, v in metrics.items()}
	wandb.log(metrics)
	_ = gc.collect()

	output = model.predict(x_test, batch_size=configs['training']['batch_size'])
	p_c_z = output['p_c_z']
	_ = gc.collect()
	if configs['globals']['save_model']:
		with open(experiment_path / 'c_test.npy', 'wb') as save_file:
			np.save(save_file, p_c_z)
	yy = np.squeeze(np.argmax(p_c_z, axis=-1))

	# Determine indeces of samples that fall into each leaf for DP&LP
	leaves = compute_leaves(model.tree)
	ind_samples_of_leaves = []
	for i in range(len(leaves)):
		ind_samples_of_leaves.append([leaves[i]['node'],np.where(yy==i)[0]])
	# Calculate leaf and dedrogram purity
	dp = utils.dendrogram_purity(model.tree, y_test, ind_samples_of_leaves)
	lp = utils.leaf_purity(model.tree, y_test, ind_samples_of_leaves)
	# Note: Only comparable DP & LP values wrt baselines if same n_leaves for all methods
	wandb.log({"Test Dendrogram Purity": dp, "Test Leaf Purity": lp})

	# Calculate confusion matrix, accuracy and nmi
	acc, idx = utils.cluster_acc(y_test, yy, return_index=True)
	swap = dict(zip(range(len(idx)), idx))
	y_wandb = np.array([swap[i] for i in yy], dtype=np.uint8)
	wandb.log({"Test_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
																	y_true=y_test, preds=y_wandb,
																	class_names=range(len(idx)))})
	nmi = normalized_mutual_info_score(y_test, yy)
	ari = adjusted_rand_score(y_test, yy)

	data_tree = construct_data_tree(model, y_predicted=yy, y_true=y_test, n_leaves=len(output['node_leaves']),
									data_name=configs['data']['data_name'])

	if configs['globals']['save_model']:
		with open(experiment_path / 'data_tree.npy', 'wb') as save_file:
			np.save(save_file, data_tree)
		with open(experiment_path / 'config.yaml', 'w', encoding='utf8') as outfile:
			yaml.dump(configs, outfile, default_flow_style=False, allow_unicode=True)

	table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree)
	fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
	dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
	wandb.log({"dendogram_final": dendro})

	wandb.log({"Test Accuracy": acc, "Test Normalized Mutual Information": nmi, "Test Adjusted Rand Index": ari})
	wandb.log({"Digits": np.unique(y_train)})
	wandb.log({"ex_name": ex_name})
	f.write("Test  |   Accuracy: %.3f, NMI: %.3f, ARI: %.3f. \n" % (acc, nmi, ari))
	print(np.unique(yy, return_counts=True))
	f.close()

	print("Accuracy:", acc)
	print("Normalized Mutual Information:", nmi)
	print("Adjusted Rand Index", ari)
	print("Digits", np.unique(y_train))

	# Compute the log-likehood
	if configs['training']['compute_ll']:
		print('\nComputing the log likelihood.... it might take a while.')
		ESTIMATION_SAMPLES = 1000
		elbo = np.zeros((len(x_test), ESTIMATION_SAMPLES))
		for j in range(ESTIMATION_SAMPLES):
			res = model.predict(x_test, batch_size=1024)
			elbo[:, j] = res['elbo_samples']
			_ = gc.collect()
		elbo_new = elbo[:, :ESTIMATION_SAMPLES]
		log_likel = np.log(1 / ESTIMATION_SAMPLES) + scipy.special.logsumexp(-elbo_new, axis=1)
		marginal_log_likelihood = np.sum(log_likel) / len(x_test)
		wandb.log({"test log-likelihood": marginal_log_likelihood})
		print("Test log-likelihood", marginal_log_likelihood)

	wandb.finish(quiet=True)
