import os
import csv
import random
import argparse
import numpy as np
from PIL import Image
from skimage.transform import resize

def CoordPlot(image_dir, coord_file, nplot=None, save_w=4000, save_h=3000, tile_size=100, makegrid=1, random_select=0, plotfile='coordplot.png'):
    """
    Plot individual images as tiles according to provided coordinates
    """
    
    # read data
    coords = np.genfromtxt(coord_file, delimiter=',')
    filenames = sorted(os.listdir(image_dir))

    if nplot==None:
       nplot = len(filenames)

    # subsample if necessary
    if nplot < len(filenames):
        if random_select:
            smpl = random.sample(range(len(filenames)), nplot)
        else:
            smpl = [i for i in range(nplot)]
        filenames = [filenames[s] for s in smpl]
        coords = coords[smpl,:]
    
    # min-max tsne coordinate scaling
    for i in range(2):
        coords[:,i] = coords[:,i] - coords[:,i].min()
        coords[:,i] = coords[:,i] / coords[:,i].max()
    tx = coords[:,0]
    ty = coords[:,1]

    full_image = Image.new('RGB', (save_w, save_h))
    for fn, x, y in zip(filenames, tx, ty):
        img = Image.open(os.path.join(image_dir, fn)) 	# load raw png image
        npi = np.array(img, np.uint8)					# convert to uint np arrat
        rsz = resize(npi, (tile_size, tile_size),						# resize, which converts to float64
                    mode='constant', 
                    anti_aliasing=True)
        npi = (2**8) * rsz / rsz.max() 					# rescale back up to original 8 bit res, with max
        npi = npi.astype(np.uint8) 						# recast as uint8
        img = Image.fromarray(npi) 						# convert back to image
        full_image.paste(img, (int((save_w - tile_size) * x), int((save_h - tile_size) * y)))

    full_image.save(plotfile)


    def Cloud2Grid(coords, gridw, gridh):
        """ convert points into a grid
        """
        nx = coords[:,0]
        ny = coords[:,1]
        nx = nx - nx.min()
        ny = ny - ny.min()
        nx = gridw * nx // nx.max()
        ny = gridh * ny // ny.max()
        nc = np.column_stack((nx, ny))
        return(nc)
               

    if makegrid:
        grid_assignment = Cloud2Grid(coords, gridw=save_w/tile_size, gridh=save_h/tile_size)
        grid_assignment = grid_assignment * tile_size

        grid_image = Image.new('RGB', (save_w, save_h))
        for img, grid_pos in zip(filenames, grid_assignment):
            x, y = grid_pos
            tile = Image.open(os.path.join(image_dir, img))
            tile = tile.resize((tile_size, tile_size), Image.ANTIALIAS)
            grid_image.paste(tile, (int(x), int(y)))

        grid_image.save(plotfile[:len(plotfile)-4] + '_grid' + plotfile[-4:])



if __name__ == '__main__':
		parser = argparse.ArgumentParser(description='scatter images to coordinate pairs')
		parser.add_argument('--image_dir',  type=str, default='images', help='input image directory of png files')
		parser.add_argument('--coord_file', type=str, default='coords.csv', help='coordinate file, csv')
		parser.add_argument('--nplot',      type=int, default=None,  help='number of images to plot')
		parser.add_argument('--save_w',     type=int, default=4000, help='width of saved image')
		parser.add_argument('--save_h',     type=int, default=3000, help='height of saved image')
		parser.add_argument('--tile_size',  type=int, default=100,  help='size of tile')
		parser.add_argument('--makegrid',   type=int, default=1,    help='raster to grid? 1=yes, 0=no')
		parser.add_argument('--random_select',     type=int, default=0,    help='random choice of images? 1=yes')
		parser.add_argument('--plotfile',   type=str, default='coordplot.png', help='name of output file')
		args = parser.parse_args()

		CoordPlot(args.image_dir, args.coord_file, args.nplot, args.save_w, 
		          args.save_h, args.tile_size, args.makegrid, args.random_select, args.plotfile)


