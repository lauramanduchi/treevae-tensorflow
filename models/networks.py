"""
Encoder, decoder, transformation, router, and dense layer architectures.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from keras import layers
import numpy as np

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


class EncoderSmall(layers.Layer):
    def __init__(self, encoded_size):
        super(EncoderSmall, self).__init__(name='enc')
        self.dense1 = tfkl.Dense(512, activation=None)
        self.bn1 = tfkl.BatchNormalization()
        self.dense2 = tfkl.Dense(512, activation=None)
        self.bn2 = tfkl.BatchNormalization()
        self.dense3 = tfkl.Dense(256, activation=None)
        self.bn3 = tfkl.BatchNormalization()
        self.dense4 = tfkl.Dense(128, activation=None)
        self.bn4 = tfkl.BatchNormalization()
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation='softplus')

    def call(self, inputs, training):
        x = self.dense1(inputs)
        x = self.bn1(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense2(x)
        x = self.bn2(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense3(x)
        x = self.bn3(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense4(x)
        x = self.bn4(x, training)
        x = tfkl.LeakyReLU()(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return x, mu, sigma

class DecoderSmall(layers.Layer):
    def __init__(self, input_shape, activation):
        super(DecoderSmall, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(128, activation=None)
        self.bn1 = tfkl.BatchNormalization()
        self.dense2 = tfkl.Dense(256, activation=None)
        self.bn2 = tfkl.BatchNormalization()
        self.dense3 = tfkl.Dense(512, activation=None)
        self.bn3 = tfkl.BatchNormalization()
        self.dense4 = tfkl.Dense(512, activation=None)
        self.bn4 = tfkl.BatchNormalization()
        if activation == "sigmoid":
            self.dense5 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense5 = tfkl.Dense(self.inp_shape)

    def call(self, inputs, training):
        x = self.dense1(inputs)
        x = self.bn1(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense2(x)
        x = self.bn2(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense3(x)
        x = self.bn3(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense4(x)
        x = self.bn4(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense5(x)
        return x

class EncoderSmallCnn(layers.Layer):
    def __init__(self, encoded_size):
        super(EncoderSmallCnn, self).__init__(name='enc')
        self.cnn0 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, activation=None)
        self.cnn1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation=None)
        self.cnn2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation=None)
        self.bn0 = tfkl.BatchNormalization()
        self.bn1 = tfkl.BatchNormalization()
        self.bn2 = tfkl.BatchNormalization()
        self.bn3 = tfkl.BatchNormalization()
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation='softplus')

    def call(self, inputs, training):
        inputs = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(inputs)
        x = self.cnn0(inputs)
        x = self.bn0(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.cnn1(x)
        x = self.bn1(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.cnn2(x)
        x = self.bn2(x, training)
        x = tfkl.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return x, mu, sigma


class DecoderSmallCnn(layers.Layer):
    def __init__(self, input_shape, activation):
        super(DecoderSmallCnn, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense = tf.keras.layers.Dense(units=3 * 3 * 32, activation=None)
        self.bn = tfkl.BatchNormalization()
        self.bn1 = tfkl.BatchNormalization()
        self.cnn1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2,
                                                    activation=None)
        self.cnn2 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',
                                                    activation=None)
        self.bn2 = tfkl.BatchNormalization()

        if activation == "sigmoid":
            self.cnn5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, activation="sigmoid", padding='same')
        else:
            self.cnn5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')

    def call(self, inputs, training):
        x = self.dense(inputs)
        x = self.bn(x, training)
        x = tfkl.LeakyReLU()(x)
        x = tf.keras.layers.Reshape(target_shape=(3, 3, 32))(x)
        x = self.cnn1(x)
        x = self.bn1(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.cnn2(x)
        x = self.bn2(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.cnn5(x)
        x = tf.keras.layers.Flatten()(x)
        return x


class EncoderOmniglot(layers.Layer):
    def __init__(self, encoded_size):
        super(EncoderOmniglot, self).__init__(name='enc')
        self.cnns = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation=None, padding='same'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, activation=None, padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=128, kernel_size=4, activation=None, padding='same'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=None),
        ]
        self.bns = [
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
        ]
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation='softplus')

    def call(self, inputs, training):
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(inputs)
        for i in range(len(self.cnns)):
            x = self.cnns[i](x)
            x = self.bns[i](x, training)
            x = tfkl.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return x, mu, sigma
        
class DecoderOmniglot(layers.Layer):
    def __init__(self, input_shape, activation):
        super(DecoderOmniglot, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense = tf.keras.layers.Dense(units=2 * 2 * 128, activation=None)
        self.cnns = [
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, activation=None, padding='same'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation=None, padding='same'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation=None)
        ]
        if activation == "sigmoid":
            self.cnns.append(tf.keras.layers.Conv2D(filters=1, kernel_size=4, padding='same', activation="sigmoid"),)
        else:
            self.cnns.append(tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=2, padding='same'))
        self.bns = [
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
            tfkl.BatchNormalization(),
        ]
        self.bn = tfkl.BatchNormalization()
        

    def call(self, inputs, training):
        x = self.dense(inputs)
        x = self.bn(x, training)
        x = tfkl.LeakyReLU()(x)
        x = tf.keras.layers.Reshape(target_shape=(2, 2, 128))(x)
        for i in range(len(self.bns)):
            x = self.cnns[i](x)
            x = self.bns[i](x, training)
            x = tfkl.LeakyReLU()(x)
        x = self.cnns[-1](x)
        x = tf.keras.layers.Flatten()(x)
        return x

def actvn(x):
    out = tfkl.LeakyReLU()(x)
    return out

class ResnetBlock(tf.keras.models.Model):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock, self).__init__()

        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = tfkl.Conv2D(filters=self.fhidden, kernel_size=3, strides=(1, 1), padding='same')
        self.conv_1 = tfkl.Conv2D(filters=self.fout, kernel_size=3, strides=(1, 1), padding='same', use_bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = tfkl.Conv2D(filters=self.fout, kernel_size=1, strides=(1, 1), padding='valid', use_bias=False)
        self.bn0 = tfkl.BatchNormalization()
        self.bn1 = tfkl.BatchNormalization()

    def call(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(self.bn0(x)))
        dx = self.conv_1(actvn(self.bn1(dx)))
        out = x_s + 0.1 * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class Resnet_Encoder(tf.keras.models.Model):
    def __init__(self, s0=2, nf=8, nf_max=256, ndim=10, size=32):
        super(Resnet_Encoder, self).__init__()

        self.s0 = s0 
        self.nf = nf  
        self.nf_max = nf_max
        self.size = size

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                tfkl.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same'),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = tfkl.Conv2D(filters=1 * nf, kernel_size=3, padding='same')

        self.resnet = tf.keras.Sequential(blocks)

        self.bn0 = tfkl.BatchNormalization()
        self.mu = tfkl.Dense(ndim, activation=None)
        self.sigma = tfkl.Dense(ndim, activation='softplus')

    def call(self, x, training=True):
        out = self.conv_img(tfkl.Reshape(target_shape=(self.size, self.size, 3))(x))
        out = self.resnet(out)
        out = tf.keras.layers.Flatten()(actvn(self.bn0(out)))
        mu = self.mu(out)
        sigma = self.sigma(out)
        return out, mu, sigma

class Resnet_Decoder(tf.keras.models.Model):
    def __init__(self, s0=2, nf=8, nf_max=256, activation='sigmoid', size=32):
        super(Resnet_Decoder, self).__init__()

        self.s0 = s0
        self.nf = nf  
        self.nf_max = nf_max 
        self.activation = activation

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        self.fc = tfkl.Dense(self.nf0 * s0 * s0, activation=None)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                tfkl.UpSampling2D(size=(2, 2), interpolation='bilinear')
            ]
        blocks += [
            ResnetBlock(nf, nf),
        ]
        self.resnet = tf.keras.Sequential(blocks)

        self.bn0 = tfkl.BatchNormalization()
        if activation == "sigmoid":
            self.conv_img = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, padding='same', activation="sigmoid")
        else:
            self.conv_img = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, padding='same')

    def call(self, z, training=True):
        out = tfkl.Reshape(target_shape=(self.s0, self.s0, self.nf0))(self.fc(z))
        out = self.resnet(out)
        out = self.conv_img(actvn(self.bn0(out)))
        out = tf.keras.layers.Flatten()(out)
        return out


# Small branch transformation
class MLP(layers.Layer):
    def __init__(self, encoded_size, hidden_unit=50):
        super(MLP, self).__init__(name='MLP')
        self.layers = layers
        self.dense1 = tfkl.Dense(hidden_unit, activation=None)
        self.bn1 = tfkl.BatchNormalization()
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation='softplus')

    def call(self, inputs, training):
        x = self.dense1(inputs)
        x = self.bn1(x, training)
        x = tfkl.LeakyReLU()(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return x, mu, sigma


class Dense(layers.Layer):
    def __init__(self, encoded_size):
        super(Dense, self).__init__(name='Dense')
        self.dense = tfkl.Dense(128, activation=None)
        self.bn = tfkl.BatchNormalization()
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation='softplus')

    def call(self, inputs, training):
        x = inputs
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Router(layers.Layer):
    def __init__(self, hidden_units=128):
        super(Router, self).__init__(name='router')
        self.dense1 = tfkl.Dense(hidden_units, activation=None)
        self.dense2 = tfkl.Dense(hidden_units, activation=None)
        self.bn1 = tfkl.BatchNormalization()
        self.bn2 = tfkl.BatchNormalization()
        self.dense3 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training, return_last_layer = False):
        x = self.dense1(inputs)
        x = self.bn1(x, training)
        x = tfkl.LeakyReLU()(x)
        x = self.dense2(x)
        x = self.bn2(x, training)
        x = tfkl.LeakyReLU()(x)
        d = self.dense3(x)
        if return_last_layer:
            return d, x
        else:
            return d


def get_encoder(architecture, encoded_size, hidden_layer, size=None):
    if architecture == 'mlp':
        encoder = EncoderSmall(encoded_size)
    elif architecture == 'cnn1':
        encoder = EncoderSmallCnn(encoded_size)
    elif architecture == 'cnn2':
        encoder = Resnet_Encoder(s0=4, nf=32, nf_max=256, ndim=encoded_size, size=size)
    elif architecture == 'cnn3':
        encoder = Resnet_Encoder(s0=4, nf=32, nf_max=256, ndim=encoded_size, size=size)
    elif architecture == 'cnn_omni':
        encoder = EncoderOmniglot(encoded_size)
    else:
        raise ValueError('The encoder architecture is mispecified.')
    return encoder


def get_decoder(architecture, hidden_layer, input_shape, activation):
    if architecture == 'mlp':
        decoder = DecoderSmall(input_shape, activation)
    elif architecture == 'cnn1':
        decoder = DecoderSmallCnn(input_shape, activation)
    elif architecture == 'cnn2':
        size = int(tf.sqrt(input_shape/3))
        decoder = Resnet_Decoder(s0=4, nf=32, nf_max=256, activation=activation, size=size)
    elif architecture == 'cnn3':
        size = int(tf.sqrt(input_shape/3))
        decoder = Resnet_Decoder(s0=4, nf=16, nf_max=256, activation=activation, size=size)
    elif architecture == 'cnn_omni':
        decoder = DecoderOmniglot(input_shape, activation) 
    else:
        raise ValueError('The decoder architecture is mispecified.')
    return decoder
