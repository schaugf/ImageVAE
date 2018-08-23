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
    
    def __init__(self, data_dir, batch_size, image_size, image_channel,
                 image_res, 
                 #channels_to_use, channel_first, 
                 shuffle):
        self.image_size = image_size
        self.batch_size = batch_size
        self.list_IDs = glob.glob(os.path.join(data_dir, 'train', '*'))
        self.image_channel = image_channel
        self.image_res = image_res
        self.shuffle = shuffle
        self.on_epoch_end()
        #self.channels_to_use = channels_to_use
        #self.channel_first = channel_first


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
        X = np.zeros((self.batch_size, self.image_size, self.image_size, self.image_channel), dtype=np.float32)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #if self.channel_first:
             #   temp = np.transpose(np.load(ID), (1, 2, 0)) / (2 ** self.image_res - 1)
           # else:
            temp = np.load(ID) / (2 ** self.image_res - 1)
            #channels = np.array(self.channels_to_use.split(',')).astype(int)
            channels = np.array(range(self.image_channel))
            temp = temp[:, :, channels]
            X[i,] = temp
            return X