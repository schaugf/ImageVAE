import os
import imageio
import glob
from PIL import Image
import numpy as np
from scipy.stats import norm
from keras.callbacks import Callback

class VAEcallback(Callback):
    """ This callback saves sample input images, their reconstructions, and a 
    latent space walk at the end of each epoch for the ImageVAE model.
    This callback accepts an ImageVAE model as its input, including both 
    encoder and decoder networks.
    
    #   Example
        '''python
            vaecb = VAEcallback(self)
        '''
    
    #   Arguments
        num_save: number of individual images (nxn) to save to grid
        vae: variational autoencoding model to generate reconstructions
        decoder: VAE decoder network to generate latent space walk visual
    
    """
    
    def __init__(self, model):
        
        self.latent_dim     = model.latent_dim
        self.latent_samp    = model.latent_samp
        self.batch_size     = model.batch_size
        self.image_size     = model.image_size
        self.num_save       = model.num_save
        self.image_channel  = model.image_channel
        self.image_res      = model.image_res
        self.data_dir       = model.data_dir
        self.save_dir       = model.save_dir
        self.vae            = model.vae
        self.decoder        = model.decoder 
        
    
    def save_input_images(self):
        """ save input images
        """
        input_figure = np.zeros((self.image_size * self.num_save, 
                         self.image_size * self.num_save, 
                         self.image_channel))
        
        to_load = glob.glob(os.path.join(self.data_dir, 'train', '*'))[:(self.num_save * self.num_save)]
        
        input_images = np.array([np.array(Image.open(fname)) for fname in to_load])
        if self.image_channel == 1:
            input_images = input_images[..., None]  # add extra index dimension

        idx = 0
        for i in range(self.num_save):
            for j in range(self.num_save):
                input_figure[i * self.image_size : (i+1) * self.image_size,
                             j * self.image_size : (j+1) * self.image_size, :] = input_images[idx,:,:,:]
                idx += 1
        
        imageio.imwrite(os.path.join(self.save_dir, 'input_images.png'),
                        input_figure.astype(np.uint8))
        
    
    def save_input_reconstruction(self, epoch):
        """ save grid of both input and reconstructed images side by side
        """
        
        recon_figure = np.zeros((self.image_size * self.num_save,
                                 self.image_size * self.num_save,
                                 self.image_channel))
        
        to_load = glob.glob(os.path.join(self.data_dir, 'train', '*'))[:(self.num_save * self.num_save)]
        
        input_images = np.array([np.array(Image.open(fname)) for fname in to_load])
        scaled_input = input_images / float((2**self.image_res - 1))
        if self.image_channel == 1:
            scaled_input = scaled_input[..., None]
       
        recon_images = self.vae.predict(scaled_input, batch_size = self.batch_size)
        scaled_recon = recon_images * float((2**self.image_res - 1))
#        scaled_recon = scaled_recon[..., None]
        
        idx = 0
        for i in range(self.num_save):
            for j in range(self.num_save):
                recon_figure[i * self.image_size : (i+1) * self.image_size,
                             j * self.image_size : (j+1) * self.image_size, :] = scaled_recon[idx,:,:,:]
                idx += 1

        imageio.imwrite(os.path.join(self.save_dir, 
                                     'reconstructed', 
                                     'recon_images_epoch_{0:03d}.png'.format(epoch)),
                        recon_figure.astype(np.uint8))
    
    
    def latent_walk(self, epoch):
        """ latent space walking
        """
        
        figure = np.zeros((self.image_size * self.latent_dim, self.image_size * self.latent_samp, self.image_channel))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, self.latent_samp))
        
        for i in range(self.latent_dim):
            for j, xi in enumerate(grid_x):
                z_sample = np.zeros(self.latent_dim)
                z_sample[i] = xi
        
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                x_decoded = self.decoder.predict(z_sample, batch_size=self.batch_size)
                x_decoded = x_decoded * float((2**self.image_res - 1))
                
                sample = x_decoded[0].reshape(self.image_size, self.image_size, self.image_channel)
                
                figure[i * self.image_size: (i + 1) * self.image_size,
                       j * self.image_size: (j + 1) * self.image_size, :] = sample
        
        imageio.imwrite(os.path.join(self.save_dir, 'latent_walk', 'latent_walk_epoch_{0:03d}.png'.format(epoch)), 
                        figure.astype(np.uint8))
        
    def on_epoch_end(self, epoch, logs={}):
        self.save_input_reconstruction(epoch)
        self.latent_walk(epoch)        

    def on_train_begin(self, logs={}):
        self.save_input_images()
        
