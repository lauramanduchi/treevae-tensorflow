# Tree Variational Autoencoders
This is the Tensorflow repository for the NeurIPS 2023 Publication (https://neurips.cc/virtual/2023/poster/71188).
Please refer to the PyTorch repository (https://github.com/lauramanduchi/treevae-pytorch) for a cleaner version of the code and for further investigation of the model performance. 

TreeVAE is a new generative method that learns the optimal tree-based posterior distribution of latent variables to capture the hierarchical structures present in the data. It adapts the architecture to discover the optimal tree for encoding dependencies between latent variables. TreeVAE optimizes the balance between shared and specialized architecture, enhancing the learning and adaptation capabilities of generative models. 
An example of a tree learned by TreeVAE is depicted in the figure below. Each edge and each split are encoded by neural networks, while the circles depict latent variables. Each sample is associated with a probability distribution over different paths of the discovered tree. The resulting tree thus organizes the data into an interpretable hierarchical structure in an unsupervised fashion, optimizing the amount of shared information between samples. In CIFAR-10, for example, the method divides the vehicles and animals into two different subtrees and similar groups (such as planes and ships) share common ancestors.

https://github.com/lauramanduchi/treevae/assets/32577028/2f473189-cb05-4482-bf77-ad128fa78b84![image](https://github.com/lauramanduchi/treevae-tensorflow/assets/32577028/1229051a-317d-4be8-9c31-e21e1591111f)


For running TreeVAE:

1. Create a new environment with the ```treevae.yml``` file.
2. Select the dataset you wish to use by changing the default config_name in the main.py parser. 
3. Potentially adapt default configuration in the config of the selected dataset (config/data_name.yml), the full set of config parameters with their explanations can be found in ```config/mnist.yml```.
4. For Weights & Biases support, set project & entity in ```train/train.py``` and change the value of ```wandb_logging``` to ```online``` in the config file.
5. Run ```main.py```.

