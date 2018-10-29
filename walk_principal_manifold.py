import os
import imageio
import itertools
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from keras.models import model_from_json

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

def WalkPrincipalManifold(model,
                          encodings,
                          output,
                          nsamples = 11):
    '''
    Walk principal manifold by inverse PCA coordinate rotation
    '''

    # testing
    #model_arch    = '/Users/schau/projects/ImageVAE/results/chc/ld16_bs16/models/arch_decoder.json'
    #model_weights = '/Users/schau/projects/ImageVAE/results/chc/ld16_bs16/models/weights_decoder.hdf5'
    #encoding_file = '/Users/schau/projects/ImageVAE/results/chc/ld16_bs16/encodings.csv'
    #nsamples      = 11
    
    os.makedirs(output, exist_ok=True)
    
    # infer plotting parameters
    image_size = model.output_shape[1]  # assumes square image

    # compute principal components of encodings
    pca = PCA(n_components=2)
    pca.fit(encodings)
    
    # generate pair-wise samples of ppd space
    norm_grid = norm.ppf(np.linspace(0.01, 
                                     0.99, 
                                     nsamples))
    
    grid_samples = list(itertools.product(norm_grid, norm_grid))
    
    inverse_grid = pca.inverse_transform(grid_samples)

    
    x_decoded = model.predict(inverse_grid, batch_size = inverse_grid.shape[0])
    
    figure = np.zeros((image_size * nsamples, 
                       image_size * nsamples, 3))
    
    counter = 0
    for i in range(nsamples):
        for j in range(nsamples):
            sample = x_decoded[counter,...]
            counter += 1
            figure[(i * image_size) : ((i + 1) * image_size),
                   (j * image_size): ((j + 1) * image_size), 
                   :] = sample * 255
        
    imageio.imwrite(os.path.join(output,
                                 'principal_manifold_walk.png'),
                    figure.astype(np.uint8))



if __name__=='__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='generate principal manifold walk')
    parser.add_argument('--model_arch', type=str, default=None, help='decoder architecture')
    parser.add_argument('--model_weights', type=str, default=None, help='decoder weights')
    parser.add_argument('--encoding_file', type=str, default=None, help='file of principal points')
    parser.add_argument('--output', type=str, default='output', help='output save image')
    parser.add_argument('--nsamples', type=int, default=11, help='number of samples')
    
    args = parser.parse_args()
    
    
    # load model from json file
    json_file = open(args.model_arch, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(args.model_weights)
    
    # load encodings
    encodings  = pd.read_csv(args.encoding_file)


    WalkPrincipalManifold(model         = loaded_model,
                          encodings     = encodings,
                          output        = args.output,
                          nsamples      = args.nsamples)


