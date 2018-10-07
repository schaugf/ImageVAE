""" Numpy data generator for keras
largely based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""

import os
import glob
import numpy as np
import keras

class NumpyDataGenerator(keras.utils.Sequence):
    """ data generator to handle .npy data objects 
    """
    
    def __init__(self, data_dir, batch_size, image_size, nchannel,
                 image_res, shuffle):
        
        self.image_size = image_size
        self.batch_size = batch_size
        self.list_IDs = glob.glob(os.path.join(data_dir, 'train', '*'))
        self.image_channel = nchannel
        self.image_res = image_res
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError()
        # Generate indexes of the batch
        # Find list of IDs
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return X, X


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.zeros((self.batch_size, self.image_size, self.image_size, self.image_channel))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(ID) / (2**self.image_res - 1)
        return X



