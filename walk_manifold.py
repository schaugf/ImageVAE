import os
import imageio
import itertools
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from keras.models import model_from_json

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


def WalkPrincipalManifold(loaded_model,
                          encodings,
                          save_dir,
                          nsamples = 11):
    '''
    Walk principal manifold by inverse PCA coordinate rotation with keras model
    '''

    
    os.makedirs(save_dir, exist_ok=True)
    
    # infer plotting parameters
    image_size = loaded_model.output_shape[1]  # assumes square image

    # compute principal components of encodings
    pca = PCA(n_components=2)
    pca.fit(encodings)
    
    # generate pair-wise samples of ppd space
    norm_grid_x = norm.ppf(np.linspace(0.01, 0.99, nsamples), scale = np.sqrt(pca.explained_variance_[0]))
    norm_grid_y = norm.ppf(np.linspace(0.01, 0.99, nsamples), scale = np.sqrt(pca.explained_variance_[1]))
    grid_samples = list(itertools.product(norm_grid_x, norm_grid_y))
    
    inverse_grid = pca.inverse_transform(grid_samples)
    
    x_decoded = loaded_model.predict(inverse_grid, batch_size = inverse_grid.shape[0])
    
    figure = np.zeros((image_size * nsamples, 
                       image_size * nsamples, 3))
    
    idx = 0
    for i in range(nsamples):
        for j in range(nsamples):
            sample = x_decoded[idx,...]
            idx += 1
            figure[(i * image_size) : ((i + 1) * image_size),
                   (j * image_size): ((j + 1) * image_size), 
                   :] = sample * 255
        
    imageio.imwrite(os.path.join(save_dir, 'walk_principal.png'), figure.astype(np.uint8))



def WalkGlobalManifold(loaded_model,
                       save_dir,
                       nsamples = 11):
    '''
    Walk global latent manifold orthogonally
    '''

    image_size = loaded_model.output_shape[1]  # assumes square image
    latent_dim = 16  # get from model

    grid_x = norm.ppf(np.linspace(0.01, 0.99, nsamples))
    sample_grid = np.zeros((latent_dim*nsamples, latent_dim))
    idx = 0
    for i in range(latent_dim):
        for j in range(nsamples):
            sample_row = np.zeros((latent_dim))
            sample_row[i] = grid_x[j]
            sample_grid[idx,:] = sample_row
            idx += 1
            
    
    x_decoded   = loaded_model.predict(sample_grid, batch_size=sample_grid.shape[0])

    figure = np.zeros((image_size * latent_dim, image_size * nsamples, 3))  # 3-channel image
    
    idx = 0
    for i in range(latent_dim):
        for j in range(nsamples):
           sample = x_decoded[idx,...] 
           idx += 1
           figure[(i * image_size) : ((i + 1) * image_size),
                  (j * image_size): ((j + 1) * image_size), :] = sample * 255
    
    imageio.imwrite(os.path.join(save_dir, 'walk_global.png'), figure.astype(np.uint8))

    

if __name__=='__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='generate principal manifold walk')
    parser.add_argument('--model_arch',    type=str, default=None, help='decoder architecture')
    parser.add_argument('--model_weights', type=str, default=None, help='decoder weights')
    parser.add_argument('--encoding_file', type=str, default=None, help='file of principal points')
    parser.add_argument('--save_dir',        type=str, default='results', help='output save image')
    parser.add_argument('--nsamples',      type=int, default=11, help='number of samples')
    
    args = parser.parse_args()
    
    
    # load model from json file
    json_file = open(args.model_arch, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(args.model_weights)
    
    # load encodings
    encodings  = pd.read_csv(args.encoding_file, header = None)


    WalkGlobalManifold(loaded_model = loaded_model,
                       save_dir     = args.save_dir,
                       nsamples     = args.nsamples)


    WalkPrincipalManifold(loaded_model  = loaded_model,
                          encodings     = encodings,
                          save_dir      = args.save_dir,
                          nsamples      = args.nsamples)


