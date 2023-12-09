import time
from pathlib import Path
import uuid
import os
import gc
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy

from utils.training_utils import AnnealKLCallback, Decay, get_optimizer
from utils.data_utils import get_data, get_gen
from utils.utils import reset_random_seeds
from baselines.lvae.model_lvae import LadderVAE

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
	os.makedirs(os.path.join(project_dir, '../../models/logs', ex_name))
	print(experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../../wandb', '.cache', 'wandb')
	wandb.init(
		project="TreeVAE_baselines",
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

	print(np.unique(y_train, return_counts=True))

	gen_train = get_gen(x_train, y_train, configs, configs['training']['batch_size'])
	gen_test = get_gen(x_test, y_test, configs, configs['training']['batch_size'], validation=True)
	# Define model & optimizer
	_ = gc.collect()
	model = LadderVAE(**configs['training'])
	optimizer = get_optimizer(configs)

	model.compile(optimizer, loss=loss)
	model.fit(gen_train, validation_data=gen_test, callbacks=[WandbCallback(), AnnealKLCallback(decay=configs['training']['decay_kl'])],
			  epochs=configs['training']['num_epochs'])
	_ = gc.collect()

	print('\n*****************model finetuning******************\n')
	model.compile(tf.keras.optimizers.Adam(learning_rate=configs['training']['lr']), loss=loss)
	model.alpha.assign(1)
	lrscheduler = Decay(lr=configs['training']['lr'], drop=0.1, epochs_drop=100).learning_rate_scheduler
	model.fit(gen_train, validation_data=gen_test,
			  callbacks=[WandbCallback(), tf.keras.callbacks.LearningRateScheduler(lrscheduler)], epochs=configs['training']['num_epochs_finetuning'])
	_ = gc.collect()

	# Save model
	if configs['globals']['save_model']:
		checkpoint_path = experiment_path
		print("\nSaving weights at ", experiment_path)
		model.save_weights(checkpoint_path)

	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)

	_ = gc.collect()

	# Test set performance
	metrics = model.evaluate(x_test, y_test, batch_size=configs['training']['batch_size'], return_dict=True)
	metrics = {f'test_{k}': v for k, v in metrics.items()}
	wandb.log(metrics)
	_ = gc.collect()
	wandb.log({"ex_name": ex_name})
	print("Digits",np.unique(y_train))

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
