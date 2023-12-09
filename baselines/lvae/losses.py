"""
Loss functions for the reconstruction term of the ELBO.
"""
import tensorflow as tf

class Losses:
    def __init__(self, configs):
        self.input_dim = configs['training']['inp_shape']
        self.tuple = False
        if isinstance(self.input_dim, list):
            print("\nData is tuple!\n")
            self.tuple = True
            self.input_dim = self.input_dim[0] * self.input_dim[1]

    def loss_reconstruction_binary(self, x, x_decoded_mean):
        loss = self.input_dim * tf.keras.losses.BinaryCrossentropy(axis=-1, reduction=tf.keras.losses.Reduction.NONE)(x, x_decoded_mean)
        return loss

    def loss_reconstruction_mse(self, x, x_decoded_mean):
        loss = self.input_dim * tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(x, x_decoded_mean)
        return loss