import os
import csv
import umap
from sklearn.manifold import TSNE

from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose
from keras import optimizers
from keras import metrics
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping

from clr_callback import CyclicLR
from vae_callback import VAEcallback
from numpydatagenerator import NumpyDataGenerator
from coordplot import CoordPlot
from walk_principal_manifold import WalkPrincipalManifold

os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 

class ImageVAE():
    """ 2-dimensional variational autoencoder for latent phenotype capture
    """
    
    def __init__(self, args):
        """ initialize model with argument parameters and build
        """

        self.data_dir       = args.data_dir
        self.image_dir      = args.image_dir
        self.save_dir       = args.save_dir    
        
        self.use_vaecb      = args.use_vaecb
        self.do_vaecb_each  = args.do_vaecb_each
        self.use_clr        = args.use_clr
        self.earlystop 		= args.earlystop
        
        self.latent_dim     = args.latent_dim
        self.inter_dim      = args.inter_dim
        self.kernel_size    = args.kernel_size
        self.batch_size     = args.batch_size
        self.epochs         = args.epochs
        self.nfilters       = args.nfilters
        self.learn_rate     = args.learn_rate
        
        self.epsilon_std    = args.epsilon_std
        
        self.latent_samp    = args.latent_samp
        self.num_save       = args.num_save
        
        self.do_tsne        = args.do_tsne
        self.verbose        = args.verbose
        self.phase          = args.phase
        self.steps_per_epoch = args.steps_per_epoch
        
        self.data_size = len(os.listdir(os.path.join(self.data_dir, 'train')))
        self.file_names = os.listdir(os.path.join(self.data_dir, 'train'))
        
        self.image_size     = args.image_size  # infer?
        self.nchannel       = args.nchannel
        self.image_res      = args.image_res
        self.show_channels  = args.show_channels
        
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
    
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    
    def build_model(self):
        """ build VAE model
        """
        
        input_shape = (self.image_size, self.image_size, self.nchannel)
        
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        
        x = inputs
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(2):
            #filters *= 2
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,
                       padding='same')(x)
        
        # shape info needed to build decoder model
        shape = K.int_shape(x)
        
        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(self.inter_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        
        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        
        for i in range(2):
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                strides=1,
                                padding='same')(x)
            #filters //= 2
        
        
        outputs = Conv2DTranspose(filters=input_shape[2],
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        
        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        plot_model(self.encoder, to_file=os.path.join(self.save_dir, 'encoder_model.png'), show_shapes=True)
        
        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file=os.path.join(self.save_dir, 'decoder_model.png'), show_shapes=True)
  
        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        #   VAE loss terms w/ KL divergence            
        def vae_loss(inputs, outputs):
            xent_loss = metrics.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
            xent_loss *= self.image_size * self.image_size
            kl_loss = 1 + z_log_var * 2 - K.square(z_mean) - K.exp(z_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss

        optimizer = optimizers.rmsprop(lr = self.learn_rate)    

        self.vae.compile(loss=vae_loss,
                         optimizer=optimizer)

        self.vae.summary()       
        plot_model(self.vae, to_file=os.path.join(self.save_dir, 'vae_model.png'), show_shapes=True)

        # save model architectures
        self.model_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        print('saving model architectures to', self.model_dir)
        with open(os.path.join(self.model_dir, 'arch_vae.json'), 'w') as file:
            file.write(self.vae.to_json())    
        with open(os.path.join(self.model_dir, 'arch_encoder.json'), 'w') as file:
            file.write(self.encoder.to_json())
        with open(os.path.join(self.model_dir, 'arch_decoder.json'), 'w') as file:
            file.write(self.decoder.to_json())
    
        
    def train(self):
        """ train VAE model
        """
        
        train_datagen = ImageDataGenerator(rescale = 1./(2**self.image_res - 1),
                                           horizontal_flip = True,
                                           vertical_flip = True)
        

        # colormode needs to be set depending on num_channels
        if self.nchannel == 1:
           train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                color_mode = 'grayscale',
                class_mode = 'input')
       
        elif self.nchannel == 3:
            print('using three channel generator!')
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.batch_size,
                color_mode = 'rgb',
                class_mode = 'input')
            
           
          
        else:
            # expecting data saved as numpy array
            train_generator = NumpyDataGenerator(self.data_dir,
                                           batch_size = self.batch_size,
                                           image_size = self.image_size,
                                           nchannel = self.nchannel,
                                           image_res = self.image_res,
                                           shuffle=True)
       
        # instantiate callbacks       
        callbacks = []

        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        csv_logger = CSVLogger(os.path.join(self.save_dir, 'training.log'), 
                               separator='\t')
        callbacks.append(csv_logger)
        
        checkpointer = ModelCheckpoint(os.path.join(self.save_dir, 'checkpoints/vae_weights.hdf5'),
                                       verbose=1, 
                                       save_best_only=True,
                                       save_weights_only=True)
        callbacks.append(checkpointer)

        if self.earlystop:
            earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
            callbacks.append(earlystop)

        if self.use_clr:
            clr = CyclicLR(base_lr=self.learn_rate,
                           max_lr=0.0001,
                           step_size=0.25*self.steps_per_epoch,
                           mode='triangular')
            callbacks.append(clr)
        
        if self.use_vaecb:
            vaecb = VAEcallback(self)
            callbacks.append(vaecb)
        
        
        self.history = self.vae.fit_generator(train_generator,
                                              epochs = self.epochs,
                                              callbacks = callbacks,
                                              steps_per_epoch = self.steps_per_epoch)                               


        print('saving model weights to', self.model_dir)
        self.vae.save_weights(os.path.join(self.model_dir, 'weights_vae.hdf5'))
        self.encoder.save_weights(os.path.join(self.model_dir, 'weights_encoder.hdf5'))
        self.decoder.save_weights(os.path.join(self.model_dir, 'weights_decoder.hdf5'))

        self.encode()

        print('done!')
   
    

    def encode(self):
        """ encode data with trained model
        """
        
        print('encoding training data...')
        test_datagen = ImageDataGenerator(rescale = 1./(2**self.image_res - 1))
        
        if self.nchannel == 1:
            test_generator = test_datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.data_size,  #1
                color_mode = 'grayscale',
                shuffle = False,
                class_mode = 'input')
            
        elif self.nchannel == 3:
            test_generator = test_datagen.flow_from_directory(
                self.data_dir,
                target_size = (self.image_size, self.image_size),
                batch_size = self.data_size,  # `
                color_mode = 'rgb',
                shuffle = False,
                class_mode = 'input')
        
        else:
          # expecting data saved as numpy array
            test_generator = NumpyDataGenerator(self.data_dir,
                                           batch_size = self.data_size,  #1
                                           image_size = self.image_size,
                                           nchannel = self.nchannel,
                                           image_res = self.image_res,
                                           shuffle=False)
       
        #encoded = self.encoder.predict_generator(test_generator,
        #                                         steps = self.data_size)
        
        encoded = self.encoder.predict_generator(test_generator)
        self.file_names = test_generator.filenames
           
         # save generated filename
        fnFile = open(os.path.join(self.save_dir, 'filenames.csv'), 'w')
        with fnFile:
            writer = csv.writer(fnFile)
            for file in self.file_names:
                writer.writerow([file])
        
        # generate and save encodings       
        outFile = open(os.path.join(self.save_dir, 'z_mean.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(encoded[0])
        
        outFile = open(os.path.join(self.save_dir, 'z_log_var.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(encoded[1])

        outFile = open(os.path.join(self.save_dir, 'encodings.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(encoded[2])


        # generate principal manifold walks
        WalkPrincipalManifold(self.decoder,
                              encoded[2],
                              self.save_dir)
        
        

        # dimensionality reduction and save
        print('learning umap...')
        umap_embed = umap.UMAP().fit_transform(encoded[2])
        outFile = open(os.path.join(self.save_dir, 'embedding_umap.csv'), 'w')
        
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(umap_embed)

        # generate coordconv figures
        CoordPlot(image_dir=self.image_dir,
                  coord_file=os.path.join(self.save_dir, 'embedding_umap.csv'),
                  save_w=8000, save_h=8000, tile_size=self.image_size,
                  plotfile=os.path.join(self.save_dir, 'coordplot_umap.png'))
       
 
        if self.do_tsne:
            print('learning tsne...')
            tsne_embed = TSNE(n_components=2).fit_transform(encoded[2])
            outFile = open(os.path.join(self.save_dir, 'embedding_tsne.csv'), 'w')
            
            with outFile:
                writer = csv.writer(outFile)
                writer.writerows(tsne_embed)

            CoordPlot(image_dir=self.image_dir,
                      coord_file=os.path.join(self.save_dir, 'embedding_tsne.csv'),
                      save_w=8000, save_h=8000, tile_size=self.image_size,
                      plotfile=os.path.join(self.save_dir, 'coordplot_tsne.png'))
 

        # external system call for plot generation
        print('generating plots with R...')
        os.system('Rscript make_plots.R -d ' + self.save_dir) 
        
        print('done!')


