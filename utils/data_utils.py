"""
Utility functions for data loading.
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_probability as tfp
import numpy as np
from utils.utils import reset_random_seeds
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from PIL import Image
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import os
import imageio

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def download_omniglot(path):
    tfds.load(
        'omniglot',
        data_dir=path,
        split='train',
        batch_size=-1,
        as_supervised=False,
    )

    tfds.load(
        'omniglot',
        data_dir=path,
        split='test',
        batch_size=-1,
        as_supervised=False,
    )
    
def get_omniglot_alphabets():
    return [
            'Alphabet_of_the_Magi',
            'Angelic',
            'Anglo-Saxon_Futhorc',
            'Arcadian',
            'Armenian',
            'Asomtavruli_(Georgian)',
            'Atemayar_Qelisayer',
            'Atlantean',
            'Aurek-Besh',
            'Avesta',
            'Balinese',
            'Bengali',
            'Blackfoot_(Canadian_Aboriginal_Syllabics)',
            'Braille',
            'Burmese_(Myanmar)',
            'Cyrillic',
            'Early_Aramaic',
            'Futurama',
            'Ge_ez',
            'Glagolitic',
            'Grantha',
            'Greek',
            'Gujarati',
            'Gurmukhi',
            'Hebrew',
            'Inuktitut_(Canadian_Aboriginal_Syllabics)',
            'Japanese_(hiragana)',
            'Japanese_(katakana)',
            'Kannada',
            'Keble',
            'Korean',
            'Latin',
            'Malay_(Jawi_-_Arabic)',
            'Malayalam',
            'Manipuri',
            'Mkhedruli_(Georgian)',
            'Mongolian',
            'N_Ko',
            'Ojibwe_(Canadian_Aboriginal_Syllabics)',
            'Old_Church_Slavonic_(Cyrillic)',
            'Oriya',
            'Sanskrit',
            'Sylheti',
            'Syriac_(Estrangelo)',
            'Syriac_(Serto)',
            'Tagalog',
            'Tengwar',
            'Tibetan',
            'Tifinagh',
            'ULOG',
    ]

def get_selected_omniglot_alphabets():    
    return np.array([13,19,39,40,11])


class DataGenSmallTree(tf.keras.utils.Sequence):

    def __init__(self, X, y, z=False, prob=False, batch_size=32, shuffle=True, augment=False, augmentation_method=False, model=False, ind_leaf=False, dataset=None):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.y_shuffled = y
        self.z = z
        self.p = prob
        self.model = model
        self.ind_leaf = ind_leaf
        self.augment = augment
        self.augmentation_method = augmentation_method
        self.shuffle = shuffle
        self.dataset = dataset
        if self.augment:
            self.shuffle = True
            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
            counter = tf.data.experimental.Counter().shuffle(1000)
            dataset = tf.data.Dataset.zip((dataset, (counter,counter)))
            AUTOTUNE = tf.data.AUTOTUNE
            if self.augmentation_method == ['simple']:
                if self.dataset == 'celeba':
                    self.data_gen = dataset.map(lambda i, s: (augment_celeba((i[0],s)),i[1]), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE) 
                else:
                    assert self.dataset in ['cifar10','cifar100']
                    self.data_gen = dataset.map(lambda i, s: (augment_cifar((i[0],s)),i[1]), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE) 
            elif self.augmentation_method == 'omniglot':
                configs = {'data': {},'training':{}}
                configs['data']['num_clusters_data'] = None
                configs['training']['augmentation_method'] = None
                configs['training']['augment'] = True
                configs['data']['data_name']='omniglot'
                self.data_gen = get_gen(X, y, configs, batch_size, validation=False)
                self.augmentation_method = ['simple']
            else:
                if self.dataset == 'celeba':
                    self.data_gen = dataset.map(lambda i, s: (augment_celeba((i[0],s)),i[1]), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE) 
                else:
                    assert self.dataset in ['cifar10','cifar100']
                    self.data_gen = dataset.map(lambda i, s: (augment_cifar((i[0],s)),i[1]), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE) 
                # Dataloader returns X' & X'' per batch
                self.batch_size = self.batch_size//2
        self.on_epoch_end()

    def on_epoch_end(self):            
        if self.augment:
            if self.augmentation_method == ['simple']:
                self.model.return_x = True
                output = self.model.predict(self.data_gen, verbose=2)
                self.model.return_x = False
                node_leaves = output['node_leaves'][self.ind_leaf]
                self.X_aug = output['input']
                self.y_shuffled = self.y
                self.z = node_leaves['z_sample']
                self.p = node_leaves['prob']
            elif len([i for i in self.augmentation_method if i in ['InfoNCE']])>0:
                self.model.return_x = True
                output1 = self.model.predict(self.data_gen, verbose=2)
                output2 = self.model.predict(self.data_gen, verbose=2)
                self.model.return_x = False
                node_leaves1 = output1['node_leaves'][self.ind_leaf]
                z1 = node_leaves1['z_sample']
                p1 = node_leaves1['prob']
                node_leaves2 = output2['node_leaves'][self.ind_leaf]
                z2 = node_leaves2['z_sample']
                p2 = node_leaves2['prob']
                self.X_aug = np.stack([output1['input'],output2['input']],1)
                self.y_shuffled = self.y
                self.z = np.stack([z1,z2],1)
                self.p = np.stack([p1,p2],1)
            else: raise NotImplementedError

        if self.shuffle:
            inds = np.arange(len(self.X))
            np.random.shuffle(inds)

            if self.augment:
                self.X_aug = self.X_aug[inds]
                self.z = self.z[inds]
                self.p = self.p[inds]
                # Keep order of y in dataloader
                self.y_shuffled = self.y[inds]
            else:
                self.X = self.X[inds]
                self.z = self.z[inds]
                self.p = self.p[inds]
                self.y = self.y[inds]  
        _ = gc.collect()     

    def __getitem__(self, index):
        if self.augment == False:
            X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
            y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        else: 
            X = self.X_aug[index * self.batch_size:(index + 1) * self.batch_size]
            y = self.y_shuffled[index * self.batch_size:(index + 1) * self.batch_size]
        z = self.z[index * self.batch_size:(index + 1) * self.batch_size]
        p = self.p[index * self.batch_size:(index + 1) * self.batch_size]
        

        if self.augment and len([i for i in self.augmentation_method if i in ['InfoNCE']])>0:
            X = tf.concat([X[:,0,:],X[:,1,:]],0)
            z = tf.concat([z[:,0,:],z[:,1,:]],0)
            p = tf.concat([p[:,0],p[:,1]],0)
            y = tf.tile(y,[2])
        return (X, z, p), y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __len__(self):
        return len(self.X) // self.batch_size
    
    
def augment_cifar_samplewise(input):
    # calculate two augmentations of same sample
    X, seed = input
    X = tf.tile(X,[2,1])
    seed1, seed2 = seed
    seed1 = tf.concat([seed1,seed1+1000],0)
    seed2 = tf.concat([seed2,seed2+1001],0)
    seed = tf.tuple([seed1,seed2])
    X_aug = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(augment_cifar,(X,seed),swap_memory=True,fn_output_signature=tf.float32))
    return X_aug

def augment_celeba_samplewise(input):
    # calculate two augmentations of same sample
    X, seed = input
    X = tf.tile(X,[2,1])
    seed1, seed2 = seed
    seed1 = tf.concat([seed1,seed1+1000],0)
    seed2 = tf.concat([seed2,seed2+1001],0)
    seed = tf.tuple([seed1,seed2])
    X_aug = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(augment_celeba,(X,seed),swap_memory=True,fn_output_signature=tf.float32))
    return X_aug



def get_gen(X, y, configs, batch_size, validation=False):
    augment = configs['training']['augment']
    augmentation_method = configs['training']['augmentation_method']
    if configs['data']['data_name']=='omniglot' and augment and not validation:
        gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=2.,
            height_shift_range=2.,
            shear_range=0.01,
            zoom_range=0.1,
            fill_mode='constant',
            cval=1.0,
        )
        X = X.reshape(-1, 28, 28,1)
        data_gen = gen.flow(
            X, 
            y,
            batch_size=batch_size,
            ignore_class_split=True,
            subset='training',
        )
    elif configs['data']['data_name']=='celeba' and augment and not validation:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        counter = tf.data.experimental.Counter().shuffle(1000)
        dataset = tf.data.Dataset.zip((dataset, (counter,counter)))
        AUTOTUNE = tf.data.AUTOTUNE
        if augmentation_method == ['simple']:
            data_gen = dataset.shuffle(buffer_size=1000).map(lambda i, s: (augment_celeba((i[0],s)),i[1]), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
        else:
            batch_size = batch_size//2
            data_gen = dataset.shuffle(buffer_size=1000).batch(batch_size).map(lambda i, s: (augment_celeba_samplewise((i[0],s)),tf.tile(i[1],[2])), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    elif augment and validation == False:
        assert configs['data']['data_name'] in ['cifar10','cifar100']
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        counter = tf.data.experimental.Counter().shuffle(1000)
        dataset = tf.data.Dataset.zip((dataset, (counter,counter)))
        AUTOTUNE = tf.data.AUTOTUNE
        if augmentation_method == ['simple']:
            data_gen = dataset.shuffle(buffer_size=1000).map(lambda i, s: (augment_cifar((i[0],s)),i[1]), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
        else:
            batch_size = batch_size//2
            data_gen = dataset.shuffle(buffer_size=1000).batch(batch_size).map(lambda i, s: (augment_cifar_samplewise((i[0],s)),tf.tile(i[1],[2])), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        AUTOTUNE = tf.data.AUTOTUNE
        data_gen = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(AUTOTUNE)
    _ = gc.collect()
    return data_gen




def get_data(configs):
    if configs['data']['data_name'] == 'mnist' or configs['data']['data_name'] == 'mnist_bin':
        reset_random_seeds(configs['globals']['seed'])
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        except:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('/path/to/data.npz')
        num_clusters = configs['data']['num_clusters_data']

        # get only num_clusters digits
        digits = np.random.choice([i for i in range(10)], size=num_clusters, replace=False)
        indx_train = np.array([], dtype=int)
        indx_test = np.array([], dtype=int)
        for i in range(num_clusters):
            indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
            indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])

        np.random.shuffle(indx_train)
        np.random.shuffle(indx_test)

        y_train = y_train[indx_train]
        x_train = x_train[indx_train]
        y_test = y_test[indx_test]
        x_test = x_test[indx_test]

        # standardize data
        x_train = x_train / 255.
        x_test = x_test / 255.

        if configs['data']['data_name'] == 'mnist_bin':
            x_train = np.random.binomial(1, x_train, size=x_train.shape).astype(np.float32)
            x_test = np.random.binomial(1, x_test, size=x_test.shape).astype(np.float32)

        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

    elif configs['data']['data_name'] == 'fmnist':
        reset_random_seeds(configs['globals']['seed'])
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        except:
            x_train, y_train = load_mnist('/path/to/data', kind='train')
            x_test, y_test = load_mnist('/path/to/data', kind='t10k')

        num_clusters = configs['data']['num_clusters_data']

        # get only num_clusters digits
        digits = np.random.choice([i for i in range(10)], size=num_clusters, replace=False)
        indx_train = np.array([], dtype=int)
        indx_test = np.array([], dtype=int)
        for i in range(num_clusters):
            indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
            indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])

        np.random.shuffle(indx_train)
        np.random.shuffle(indx_test)
        y_train = y_train[indx_train]
        x_train = x_train[indx_train]
        y_test = y_test[indx_test]
        x_test = x_test[indx_test]

        # standardize data
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 28 * 28))

    elif configs['data']['data_name'] == 'news20':
        reset_random_seeds(configs['globals']['seed'])

        try:
            newsgroups_train = fetch_20newsgroups(subset='train')
            newsgroups_test = fetch_20newsgroups(subset='test')
            vectorizer = TfidfVectorizer(max_features=2000)
            x_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
            x_test = vectorizer.transform(newsgroups_test.data).toarray()
            y_train = newsgroups_train.target
            y_test = newsgroups_test.target
        except:
            loaded = np.load('/path/to/data.npz')
            x_train, x_test = loaded['x_train'], loaded['x_test']
            y_train, y_test = loaded['y_train'], loaded['y_test']

        num_clusters = configs['data']['num_clusters_data']

        # get only num_clusters digits
        digits = np.random.choice([i for i in range(20)], size=num_clusters, replace=False)
        indx_train = np.array([], dtype=int)
        indx_test = np.array([], dtype=int)
        for i in range(num_clusters):
            indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
            indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])

        np.random.shuffle(indx_train)
        np.random.shuffle(indx_test)
        y_train = y_train[indx_train]
        x_train = x_train[indx_train]
        y_test = y_test[indx_test]
        x_test = x_test[indx_test]


    elif configs['data']['data_name'] in ['cifar10', 'cifar10_vehicles', 'cifar10_animals']:
        reset_random_seeds(configs['globals']['seed'])
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        except:
            import tarfile
            import pickle as pkl
            f = tarfile.open('/path/to/cifar-10-python.tar.gz', 'r:gz')
            b1 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_1"), encoding="bytes")
            b2 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_2"), encoding="bytes")
            b3 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_3"), encoding="bytes")
            b4 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_4"), encoding="bytes")
            b5 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_5"), encoding="bytes")
            test = pkl.load(f.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
            x_train = np.concatenate([b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data']], axis=0)
            #x_train = np.asarray(x_train, dtype='float32')
            y_train = np.concatenate([np.array(b1[b'labels']),
                                    np.array(b2[b'labels']),
                                    np.array(b3[b'labels']),
                                    np.array(b4[b'labels']),
                                    np.array(b5[b'labels'])], axis=0)

            x_test = test[b'data']
            #x_test = np.asarray(x_test, dtype='float32')
            y_test = np.array(test[b'labels'])

            # Important: Cifar is stored with 3 color channels in first dim, so we need to reshape it before flattening!
            x_train = np.reshape(x_train, (-1, 3, 32, 32))
            x_test = np.reshape(x_test, (-1, 3, 32, 32))
            x_train = np.moveaxis(x_train, [1],[3])
            x_test = np.moveaxis(x_test, [1],[3])
            f.close()

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        num_clusters = configs['data']['num_clusters_data']

        digits = np.array(range(10))
        indx_train = np.array([], dtype=int)
        indx_test = np.array([], dtype=int)
        for i in range(num_clusters):
            indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
            indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])

        np.random.shuffle(indx_train)
        np.random.shuffle(indx_test)

        y_train = y_train[indx_train]
        x_train = x_train[indx_train]
        y_test = y_test[indx_test]
        x_test = x_test[indx_test]

        # standardize data
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 32 * 32 * 3))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 32 * 32 * 3))
        
        x_train = np.asarray(x_train, dtype='float32')
        x_test = np.asarray(x_test, dtype='float32')

    elif configs['data']['data_name'] == 'cifar100':

        reset_random_seeds(configs['globals']['seed'])
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse') 
        except:
            data_dir = '/path/to/dataparentfolder'
            folder_name = 'cifar-100-python'
            filenames = ['train', 'test']

            filenames = list(map(lambda x: os.path.join(data_dir, folder_name, x), filenames))
            def _read_cifar(filenames, cls):
                import pickle
                image = []
                label = []
                for fname in filenames:
                    with open(fname, 'rb') as f:
                        raw_dict = pickle.load(f, encoding='latin1')
                    raw_data = raw_dict['data']
                    label.extend(raw_dict['coarse_labels' if cls == 'coarse' else 'fine_labels'])
                    for x in raw_data:
                        x = x.reshape(3, 32, 32)
                        x = np.transpose(x, [1, 2, 0])
                        image.append(x)
                return np.array(image), np.array(label)
            train_set = _read_cifar(filenames[:-1], 'coarse')
            x_train = np.asarray(train_set[0], dtype='float32')
            y_train = np.array(train_set[1])
            test_set = _read_cifar([filenames[-1]], 'coarse')
            x_test = np.asarray(test_set[0], dtype='float32')
            y_test = np.array(test_set[1])

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        num_clusters = configs['data']['num_clusters_data']

        digits = np.array(range(num_clusters))
        
        indx_train = np.array([], dtype=int)
        indx_test = np.array([], dtype=int)
        for i in range(num_clusters):
            indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
            indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])

        np.random.shuffle(indx_train)
        np.random.shuffle(indx_test)

        y_train = y_train[indx_train]
        x_train = x_train[indx_train]
        y_test = y_test[indx_test]
        x_test = x_test[indx_test]

        # standardize data
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 32 * 32 * 3))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 32 * 32 * 3))
        
        x_train = np.asarray(x_train, dtype='float32')
        x_test = np.asarray(x_test, dtype='float32')
        

    elif configs['data']['data_name'] == 'omniglot':
        reset_random_seeds(configs['globals']['seed'])
        x = tfds.as_numpy(tfds.load(
            'omniglot',
            data_dir=configs['data']['path'],
            split='train',
            batch_size=-1,
            as_supervised=False,
        ))
        y_train = x['alphabet']
        x_train = x['image']
        train_stratify_labels = x['label']

        x = tfds.as_numpy(tfds.load(
            'omniglot',
            data_dir=configs['data']['path'],
            split='test',
            batch_size=-1,
            as_supervised=False,
        ))
        y_test = x['alphabet']
        x_test = x['image']
        test_stratify_labels = x['label']

        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        y = np.squeeze(y)
        stratify_labels = np.concatenate([train_stratify_labels, test_stratify_labels], axis=0)
        stratify_labels = np.squeeze(stratify_labels)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=configs['globals']['seed'], stratify=stratify_labels)
        
        # Resize test set
        x_train = (tf.image.resize(x_train, [28,28]).numpy()[:, :, :, 0]).astype(np.float32)/255.
        x_test = (tf.image.resize(x_test, [28,28]).numpy()[:, :, :, 0]).astype(np.float32)/255.

        num_clusters = configs['data']['num_clusters_data']

        # get only num_clusters digits
        if num_clusters !=50:
            digits = get_selected_omniglot_alphabets()[:num_clusters]
        else:
            digits = np.arange(50)
        indx_train = np.array([], dtype=int)
        indx_test = np.array([], dtype=int)
        for i in range(num_clusters):
            indx_train = np.append(indx_train, np.where(y_train == digits[i])[0])
            indx_test = np.append(indx_test, np.where(y_test == digits[i])[0])
        
        x_train, y_train = shuffle(x_train[indx_train], y_train[indx_train], random_state=configs['globals']['seed'])
        x_test, y_test = shuffle(x_test[indx_test], y_test[indx_test], random_state=configs['globals']['seed'])
        
        # Relabel values in y
        y_train = y_train+50
        y_test = y_test+50
        for idx, digit in enumerate(digits):
            y_test[y_test==digit+50] = idx    
            y_train[y_train==digit+50] = idx
        # standardize data
        x_train = np.reshape(x_train, (-1, 28 * 28 * 1))
        x_test = np.reshape(x_test, (-1, 28 * 28 * 1))

        
    elif configs['data']['data_name'] ==  'celeba':
        reset_random_seeds(configs['globals']['seed'])
        data_dir = configs['data']['path']
        dimensions = 64
        try:
            x_test  = np.load(os.path.join(data_dir, 'celeba%d'%dimensions,'test.npy'))
        except:
            print('loading test data')
            x_test  = load_celeba_data(data_dir,'test', dimensions)
            np.save(os.path.join(data_dir, 'celeba%d'%dimensions, 'test.npy'), x_test)
        try:
            x_train = np.load(os.path.join(data_dir, 'celeba%d'%dimensions,'train.npy'))
        except:
            print('loading training data')
            x_train = load_celeba_data(data_dir,'training', dimensions)
            np.save(os.path.join(data_dir, 'celeba%d'%dimensions, 'train.npy'), x_train)


        y_train = np.squeeze(np.zeros_like(x_train[:,0,0,0]))
        y_test = np.squeeze(np.zeros_like(x_test[:,0,0,0]))

        indx_train = np.arange(len(y_train))
        indx_test = np.arange(len(y_test))

        np.random.shuffle(indx_train)
        np.random.shuffle(indx_test)

        y_train = y_train[indx_train]
        x_train = x_train[indx_train]
        y_test = y_test[indx_test]
        x_test = x_test[indx_test]


        # Pick only 50k random train images & 10k test
        x_train = x_train[:100000]
        y_train = y_train[:100000]
        x_test = x_test
        y_test = y_test

        # standardize data
        x_train = x_train / 255.
        x_train = np.reshape(x_train, (-1, 64 * 64 * 3))
        x_test = x_test / 255.
        x_test = np.reshape(x_test, (-1, 64 * 64 * 3))

        x_train = np.asarray(x_train, dtype='float32')
        x_test = np.asarray(x_test, dtype='float32')
        _ = gc.collect
        
    else:
        raise NotImplementedError('This dataset is not supported!')

    return x_train, x_test, y_train, y_test


def load_celeba_data(data_dir, flag='training', side_length=None, num=None):
    dir_path = os.path.join(data_dir, 'img_align_celeba')
    filelist = [filename for filename in sorted(os.listdir(dir_path)) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imageio.imread(dir_path + os.sep + filelist[i]))
        if side_length is not None:
            img = Image.fromarray(img)
            img = img.crop((15, 40, 178 - 15, 218 - 30))
            img = np.asarray(img.resize([side_length, side_length]))
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)

def augment_cifar(input, s=0.5, brightness_factor=0.8, contrast_factor=0.8, saturation_factor=0.8, hue_factor=0.2):
    # Only takes one sample at a time
    x, seed = input
    new_images = tf.reshape(x, [1, 32, 32, 3])
    brightness_factor = brightness_factor * s
    contrast_factor = contrast_factor * s
    saturation_factor = saturation_factor * s
    hue_factor = hue_factor * s
    probs = tf.random.stateless_uniform(shape=([10]), minval=0, maxval=1, seed=seed)
    seeds = [tf.tuple(val + i for val in seed) for i in range(1010,1025)]

    # Augmentation 1  
    new_images = randomresizedcrop(new_images, seeds[0])
    # Augmentation 2
    new_images = tf.image.stateless_random_flip_left_right(new_images, seed=seeds[1])

    # Augmentation 3
    if probs[0]<0.8:
        # Manual brightness change due to implementation differences of pytorch and tensorflow
        brightness = tf.random.stateless_uniform(shape=(1,), minval=max(0, 1 - brightness_factor), maxval= 1 + brightness_factor, seed=seeds[2])
        new_images = tf.clip_by_value(new_images*brightness,0,1)
        new_images = tf.image.stateless_random_contrast(new_images, max(0, 1 - contrast_factor), 1 + contrast_factor, seed=seeds[3])
        new_images = tf.image.stateless_random_saturation(new_images, max(0, 1 - saturation_factor), 1 + saturation_factor, seed=seeds[4])
        new_images = tf.image.stateless_random_hue(new_images, hue_factor, seed=seeds[5])    
    
    # Augmentation 4
    if probs[1]<0.2:
        new_images = tf.image.rgb_to_grayscale(new_images)
        new_images = tf.image.grayscale_to_rgb(new_images)

    new_images = tf.clip_by_value(tf.reshape(new_images, [32*32*3]),0,1)
    return new_images



def augment_celeba(input, s=0.25, brightness_factor=0.8, contrast_factor=0.8, saturation_factor=0.8, hue_factor=0.2):
    # Only takes one sample at a time
    x, seed = input
    new_images = tf.reshape(x, [1, 64, 64, 3])
    brightness_factor = brightness_factor * s
    contrast_factor = contrast_factor * s
    saturation_factor = saturation_factor * s
    hue_factor = hue_factor * s
    probs = tf.random.stateless_uniform(shape=([10]), minval=0, maxval=1, seed=seed)
    seeds = [tf.tuple(val + i for val in seed) for i in range(1010,1025)]

    # Augmentation 1  
    new_images = randomresizedcrop(new_images, seeds[0], scale = (3/4,1), ratio=(4/5,5/4), crop_shape=(64,64))
    # Augmentation 2
    new_images = tf.image.stateless_random_flip_left_right(new_images, seed=seeds[1])

    # Augmentation 3
    if probs[0]<0.8:
        brightness = tf.random.stateless_uniform(shape=(1,), minval=max(0, 1 - brightness_factor), maxval= 1 + brightness_factor, seed=seeds[2])
        new_images = tf.clip_by_value(new_images*brightness,0,1)
        new_images = tf.image.stateless_random_contrast(new_images, max(0, 1 - contrast_factor), 1 + contrast_factor, seed=seeds[3])
        new_images = tf.image.stateless_random_saturation(new_images, max(0, 1 - saturation_factor), 1 + saturation_factor, seed=seeds[4])

    new_images = tf.clip_by_value(tf.reshape(new_images, [64*64*3]),0,1)
    return new_images


def randomresizedcrop(images, seed, scale = (0.2,1), ratio = (3/4,4/3),crop_shape = (32,32),batch_size=1):
    
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    random_scales = tf.random.stateless_uniform(
        (batch_size,),
        tf.tuple(val + 1 for val in seed),
        scale[0],
        scale[1]
    )
    
    random_ratios = tf.exp(tf.random.stateless_uniform(
        (batch_size,),
        tf.tuple(val + 2 for val in seed),
        log_ratio[0],
        log_ratio[1]
    ))
    
    new_heights = tf.clip_by_value(
        tf.sqrt(random_scales / random_ratios),
        0,
        1,
    )
    new_widths = tf.clip_by_value(
        tf.sqrt(random_scales * random_ratios),
        0,
        1,
    )
    height_offsets = tf.random.stateless_uniform(
        (batch_size,),
        tf.tuple(val + 3 for val in seed),
        0,
        1 - new_heights,
    )
    width_offsets = tf.random.stateless_uniform(
        (batch_size,),
        tf.tuple(val + 4 for val in seed),
        0,
        1 - new_widths,
    )

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )
    images = tf.image.crop_and_resize(
        images,
        bounding_boxes,
        tf.range(batch_size),
        crop_shape,
    )
    return images