import os
import csv
import glob 
import imageio
from PIL import Image

import numpy as np
from scipy.stats import norm

from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose
from keras import optimizers
from keras import metrics
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, Callback


class ImgSave(Callback):
    """ this callback saves sample input images, their reconstructions, and a 
    latent space walk at the end of each epoch
    """
    
    def __init__(self, latent_dim, latent_samp, batch_size, image_size, num_save, 
                 image_channel, image_res, data_dir, save_dir, vae, decoder):
        
        self.latent_dim     = latent_dim
        self.latent_samp    = latent_samp
        self.batch_size     = batch_size
        self.image_size     = image_size
        self.num_save       = num_save
        self.image_channel  = image_channel
        self.image_res      = image_res
        self.data_dir       = data_dir
        self.save_dir       = save_dir
        self.vae            = vae
        self.decoder        = decoder 
        
    
    def save_input_images(self):
        """ save input 
        """
    
    def save_input_reconstruction(self, epoch):
        """ save grid of both input and reconstructed images side by side
        """
        
        input_figure = np.zeros((self.image_size * self.num_save, 
                                 self.image_size * self.num_save, 
                                 self.image_channel))
        
        recon_figure = np.zeros((self.image_size * self.num_save,
                                 self.image_size * self.num_save,
                                 self.image_channel))
        
        to_load = glob.glob(os.path.join(self.data_dir, 'train', '*'))[:(self.num_save * self.num_save)]
        
        input_images = np.array([np.array(Image.open(fname)) for fname in to_load])
        scaled_input = input_images / float((2**self.image_res - 1))
        
        recon_images = self.vae.predict(scaled_input, batch_size = self.batch_size)
        scaled_recon = recon_images * float((2**self.image_res - 1))
        
        idx = 0
        for i in range(self.num_save):
            for j in range(self.num_save):
                input_figure[i * self.image_size : (i+1) * self.image_size,
                             j * self.image_size : (j+1) * self.image_size, :] = input_images[idx,:,:,:]
                recon_figure[i * self.image_size : (i+1) * self.image_size,
                             j * self.image_size : (j+1) * self.image_size, :] = scaled_recon[idx,:,:,:]
                idx += 1
        
        imageio.imwrite(os.path.join(self.save_dir, 
                                     'input', 
                                     'input_images_epoch_{0:03d}.png'.format(epoch)), input_figure)
        
        imageio.imwrite(os.path.join(self.save_dir, 
                                     'reconstructed', 
                                     'recon_images_epoch_{0:03d}.png'.format(epoch)), recon_figure)
    
    
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
        
        imageio.imwrite(os.path.join(self.save_dir, 'latent_walk', 'latent_walk_epoch_{0:03d}.png'.format(epoch)), figure)
        
    def on_epoch_end(self, epoch, logs={}):
        self.save_input_reconstruction(epoch)
        self.latent_walk(epoch)        

    def on_train_begin(self, logs={}):
        self.save_input()
        

class ImageVAE():
    """ 2-dimensional variational autoencoder for latent phenotype capture
    """
    
    def __init__(self, args):
        """ initialize model with argument parameters and build
        """

        self.data_dir       = args.data_dir
        self.save_dir       = args.save_dir
        
        self.image_size     = args.image_size
        self.image_channel  = args.image_channel
        self.image_res      = args.image_res
        
        self.latent_dim     = args.latent_dim
        self.inter_dim      = args.inter_dim
        self.num_conv       = args.num_conv
        self.batch_size     = args.batch_size
        self.epochs         = args.epochs
        self.nfilters       = args.nfilters
        self.learn_rate     = args.learn_rate
        self.epsilon_std    = args.epsilon_std
        self.latent_samp    = args.latent_samp
        self.num_save       = args.num_save
        self.verbose        = args.verbose
        
        self.phase          = args.phase
        
        self.steps_per_epoch = args.steps_per_epoch
        
        self.data_size = len(os.listdir(os.path.join(self.data_dir, 'train')))
        
        if self.steps_per_epoch == 0:
            self.steps_per_epoch = self.data_size // self.batch_size
                
        self.build_model()


    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """
        
        z_mean, z_log_var = sample_args
        
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                         self.latent_dim),
                                  mean=0,
                                  stddev=self.epsilon_std)
    
        return z_mean + K.exp(z_log_var) * epsilon
    
    
    def build_model(self):
        """ build VAE model
        """
        
        input_dim = (self.image_size, self.image_size, self.image_channel)
        
        #   encoder architecture
        
        x = Input(shape=input_dim)
        
        conv_1 = Conv2D(self.image_channel,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu')(x)
        
        conv_2 = Conv2D(self.nfilters,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu',
                        strides=(2, 2))(conv_1)
        
        conv_3 = Conv2D(self.nfilters,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_2)
        
        conv_4 = Conv2D(self.nfilters,
                        kernel_size=self.num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_3)
        
        flat = Flatten()(conv_4)
        hidden = Dense(self.inter_dim, activation='relu')(flat)
        
        #   reparameterization trick
        
        z_mean      = Dense(self.latent_dim)(hidden)        
        z_log_var   = Dense(self.latent_dim)(hidden)
        
        z           = Lambda(self.sampling)([z_mean, z_log_var])
        
        
        #   decoder architecture

        output_dim = (self.batch_size, 
                      self.image_size//2,
                      self.image_size//2,
                      self.nfilters)
        
        #   instantiate rather than pass through for later resuse
        
        decoder_hid = Dense(self.inter_dim, 
                            activation='relu')
        
        decoder_upsample = Dense(self.nfilters * 
                                 self.image_size//2 * 
                                 self.image_size//2, 
                                 activation='relu')

        decoder_reshape = Reshape(output_dim[1:])
        
        decoder_deconv_1 = Conv2DTranspose(self.nfilters,
                                           kernel_size=self.num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        
        decoder_deconv_2 = Conv2DTranspose(self.nfilters,
                                   kernel_size=self.num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
        
        decoder_deconv_3_upsamp = Conv2DTranspose(self.nfilters,
                                                  kernel_size = self.num_conv,
                                                  strides = (2, 2),
                                                  padding = 'valid',
                                                  activation = 'relu')
        
        decoder_mean_squash = Conv2D(self.image_channel,
                                     kernel_size = 2,
                                     padding = 'valid',
                                     activation = 'sigmoid')
        
        hid_decoded             = decoder_hid(z)
        up_decoded              = decoder_upsample(hid_decoded)
        reshape_decoded         = decoder_reshape(up_decoded)
        deconv_1_decoded        = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded        = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu          = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash   = decoder_mean_squash(x_decoded_relu)

        #   need to keep generator model separate so new inputs can be used
        
        decoder_input           = Input(shape=(self.latent_dim,))
        _hid_decoded            = decoder_hid(decoder_input)
        _up_decoded             = decoder_upsample(_hid_decoded)
        _reshape_decoded        = decoder_reshape(_up_decoded)
        _deconv_1_decoded       = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded       = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu         = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash  = decoder_mean_squash(_x_decoded_relu)
        
        #   instantiate VAE models
        
        self.vae        = Model(x, x_decoded_mean_squash)
        self.encoder    = Model(x, z_mean)
        self.decoder    = Model(decoder_input, _x_decoded_mean_squash)
        
        #   VAE loss terms w/ KL divergence
            
        def vae_loss(x, x_decoded_mean_squash):
            xent_loss = self.image_size * self.image_size * metrics.binary_crossentropy(K.flatten(x),
                                                                                        K.flatten(x_decoded_mean_squash))
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss
        
        
        adam = optimizers.adam(lr = self.learn_rate)
        
        self.vae.compile(optimizer = adam,
                         loss = vae_loss)
        
        self.vae.summary()
            
    
    def train(self):
        """ train VAE model
        """
        
        train_datagen = ImageDataGenerator(rescale = 1./(2**self.image_res - 1),
                                           horizontal_flip = True,
                                           vertical_flip = True)
                        
        train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                class_mode = 'input')
                
        # instantiate callbacks
        
        term_nan = TerminateOnNaN()
        
        csv_logger = CSVLogger(os.path.join(self.save_dir, 'training.log'), 
                               separator='\t')
        
        checkpointer = ModelCheckpoint(os.path.join(self.save_dir, 
                                                    'checkpoints/vae_weights.hdf5'), 
                                       verbose=1, 
                                       save_weights_only=True)
        
        # custom image saving callback
        img_saver = ImgSave(self.latent_dim,
                            self.latent_samp,
                            self.batch_size,
                            self.image_size, 
                            self.num_save, 
                            self.image_channel, 
                            self.image_res, 
                            self.data_dir, 
                            self.save_dir, 
                            self.vae,
                            self.decoder)
        
        self.history = self.vae.fit_generator(train_generator,
                               epochs = self.epochs,
                               verbose = self.verbose,
                               callbacks = [term_nan,
                                            csv_logger,
                                            checkpointer,
                                            img_saver],
                               steps_per_epoch = self.steps_per_epoch)                               

        self.vae.save_weights(os.path.join(self.save_dir, 
                                           'checkpoints/vae_weights.hdf5'))
        self.encode()        
        
    
    def encode(self):
        """ encode data with trained model
        """
        
        test_datagen = ImageDataGenerator(rescale = 1./(2**self.image_res - 1))
        
        test_generator = test_datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                shuffle = False,
                class_mode = 'input')
        
        print('encoding training data...')
        x_test_encoded = self.encoder.predict_generator(test_generator,
                                                        steps = self.data_size // self.batch_size)
        
        outFile = open(os.path.join(self.save_dir, 'encodings.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(x_test_encoded)
        
        
