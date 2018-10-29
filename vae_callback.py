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
        self.nchannel       = model.nchannel
        self.image_res      = model.image_res
        self.data_dir       = model.data_dir
        self.save_dir       = model.save_dir
        self.vae            = model.vae
        self.decoder        = model.decoder
        self.show_channels  = model.show_channels
        self.do_vaecb_each  = model.do_vaecb_each  # do each epoch
    
    def save_input_images(self):
        """ save input images
        """
        input_figure = np.zeros((self.image_size * self.num_save, 
                         self.image_size * self.num_save, 
                         min(3, self.nchannel))) 
        
        to_load = glob.glob(os.path.join(self.data_dir, 'train', '*'))[:(self.num_save * self.num_save)]
        
        
        if self.nchannel > 3:
            input_images = np.array([np.load(fname) for fname in to_load])
            input_images = ((2**8 - 1) * (input_images[...,self.show_channels]/(2**self.image_res - 1))).astype(np.uint8)
        
        else:
            input_images = np.array([np.array(Image.open(fname)) for fname in to_load])  # needs to generalize to 2-channel?
        
        if self.nchannel == 1:
            input_images = input_images[..., None]  # add extra index dimension

        idx = 0
        for i in range(self.num_save):
            for j in range(self.num_save):
                input_figure[i * self.image_size : (i+1) * self.image_size,
                             j * self.image_size : (j+1) * self.image_size, :] = input_images[idx,:,:,:]
                idx += 1
        
        imageio.imwrite(os.path.join(self.save_dir, 'input_images.png'),
                        input_figure.astype(np.uint8))
        
    
    def save_input_reconstruction(self, epoch=0, is_final=False):
        """ save grid of both input and reconstructed images side by side
        """
        
        recon_figure = np.zeros((self.image_size * self.num_save,
                                 self.image_size * self.num_save,
                                 min(3, self.nchannel)))
        
        to_load = glob.glob(os.path.join(self.data_dir, 'train', '*'))[:(self.num_save * self.num_save)]
        
        # again, whether input is npy or png
        if self.nchannel > 3:
            input_images = np.array([np.load(fname) for fname in to_load])
            
        else:
            input_images = np.array([np.array(Image.open(fname)) for fname in to_load])
        
        scaled_input = input_images / (2**self.image_res - 1)

        if self.nchannel == 1:
            scaled_input = scaled_input[..., None]
            
               
        recon_images = self.vae.predict(scaled_input, batch_size = self.batch_size)
        scaled_recon = recon_images * (2**8 - 1)
        scaled_recon = scaled_recon[...,self.show_channels]
        
        idx = 0
        for i in range(self.num_save):
            for j in range(self.num_save):
                #print('reconstruction max:', scaled_recon.max())
                #print('reconstructed dtype:', scaled_recon.dtype)

                recon_figure[i * self.image_size : (i+1) * self.image_size,
                             j * self.image_size : (j+1) * self.image_size, :] = scaled_recon[idx,...]
                idx += 1

        if not(is_final):
            imageio.imwrite(os.path.join(self.save_dir, 
                                         'reconstructed', 
                                         'recon_images_epoch_{0:03d}.png'.format(epoch)),
                                         recon_figure.astype(np.uint8))
        else:
            imageio.imwrite(os.path.join(self.save_dir, 
                                         'recon_images_final.png'),
                                         recon_figure.astype(np.uint8))

    
    
    def latent_walk(self, epoch=0, is_final=False):
        """ latent space walking
        """
        
        figure = np.zeros((self.image_size * self.latent_dim, self.image_size * self.latent_samp, min(3, self.nchannel)))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, self.latent_samp))
        
        for i in range(self.latent_dim):
            for j, xi in enumerate(grid_x):
                z_sample = np.zeros(self.latent_dim)
                z_sample[i] = xi
        
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                
                x_decoded = self.decoder.predict(z_sample, batch_size=self.batch_size)
                x_decoded = x_decoded * float(2**8 - 1)
                
                sample = x_decoded[0].reshape(self.image_size, self.image_size, self.nchannel)
                
                # need show_channels input as list
                if self.nchannel > 3:
                    sample = sample[:,:,self.show_channels]
                
                figure[i * self.image_size: (i + 1) * self.image_size,
                       j * self.image_size: (j + 1) * self.image_size, :] = sample
        
        if not(is_final):
            imageio.imwrite(os.path.join(self.save_dir, 'latent_walk', 'latent_walk_epoch_{0:03d}.png'.format(epoch)), 
                            figure.astype(np.uint8))
        else:
            imageio.imwrite(os.path.join(self.save_dir, 'latent_walk_final.png'), 
                            figure.astype(np.uint8))
        
        
    def on_epoch_end(self, epoch, logs={}):
        if self.do_vaecb_each:
            self.save_input_reconstruction(epoch)
            self.latent_walk(epoch)        


    def on_train_begin(self, logs={}):
        os.makedirs(os.path.join(self.save_dir, 'latent_walk'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'reconstructed'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'animated'), exist_ok=True)
        self.save_input_images()

    
    def on_train_end(self, logs={}):
        self.save_input_reconstruction(is_final=True)
        self.latent_walk(is_final=True)

        if self.do_vaecb_each:
            print('animating training...')
            cmd = 'ffmpeg -i ' + self.save_dir + '/latent_walk/latent_walk_epoch_%03d.png -vcodec libx264 -crf 25 ' + self.save_dir + '/animated/latent_walk_animated.mp4'
            os.system(cmd)
        
            cmd = 'ffmpeg -i ' + self.save_dir + '/reconstructed/recon_images_epoch_%03d.png -vcodec libx264 -crf 25 ' + self.save_dir + '/animated/reconstructed_animated.mp4'
            os.system(cmd)


