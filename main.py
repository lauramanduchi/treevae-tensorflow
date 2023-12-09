"""
Runs the treeVAE model.
"""
import argparse
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import distutils

from train.train import run_experiment
from utils.utils import prepare_config


tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.list_physical_devices(device_type = 'GPU')

try:
	tf.config.experimental.set_memory_growth(gpus[0], True)
except:
	print("Not enough GPU hardware devices available")


def main():
	project_dir = Path(__file__).absolute().parent
	print(project_dir)

	parser = argparse.ArgumentParser()

	# Model parameters
	parser.add_argument('--data_name', type=str, choices=['mnist', 'fmnist', 'news20', 'omniglot','news20', 'cifar100', 'celeba'], help='the dataset')
	parser.add_argument('--num_epochs', type=int, help='the number of training epochs')
	parser.add_argument('--num_epochs_finetuning', type=int, help='the number of finetuning epochs')
	parser.add_argument('--num_clusters_data', type=int, help='the number of vlusters of the tree')
	
	parser.add_argument('--kl_start', type=float, nargs='?', const=0., help='initial KL divergence from where annealing starts')
	parser.add_argument('--decay_kl', type=float, help='KL divergence annealing')
	parser.add_argument('--latent_dim', type=str, help='specifies the latent dimensions of the tree')
	parser.add_argument('--mlp_layers', type=str, help='specifies how many layers should the MLPs have')

	parser.add_argument('--grow', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to grow the tree')
	parser.add_argument('--grow_mode', type=int, help='grow mode')

	parser.add_argument('--compute_ll', type=lambda x: bool(distutils.util.strtobool(x)), help='whether to compute the log-likelihood')
	parser.add_argument('--augment', type=lambda x:bool(distutils.util.strtobool(x)), help='augment images or not')
	parser.add_argument('--augmentation_method', type=str, help='simply augment inputs vs contrastive-ish approaches')
	parser.add_argument('--aug_decisions_weight', type=float, help='weight of similarity regularizer for augmented images')

	# Other parameters
	parser.add_argument('--save_model', type=lambda x:bool(distutils.util.strtobool(x)), help='specifies if the model should be saved')
	parser.add_argument('--config_name', default='mnist', type=str, help='the override file name for config.yml')
	parser.add_argument('--seed', type=int, help='random number generator seed')
	parser.add_argument('--wandb_logging', type=str, help='online, disabled, offline enables logging in wandb')
	
	
	args = parser.parse_args()
	configs, loss = prepare_config(args, project_dir)
	run_experiment(configs, loss)


if __name__ == "__main__":
	main()