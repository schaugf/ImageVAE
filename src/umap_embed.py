import sys
import csv
import numpy as np
import umap

#   embedding input file

encoding_file  = sys.argv[1]
write_file     = sys.argv[2]

print('loading vae encoding file', encoding_file)
print('embedding saving to', write_file)

#   read csv embedding file

encoding_data = np.genfromtxt(encoding_file, delimiter=',')
print('embedding found with shape', encoding_data.shape)

#   run embedding

print('running umap...')
umap_embedding = umap.UMAP().fit_transform(encoding_data)
print('done')

#   write umap embedding

print('saving...')
with open(write_file, 'w') as file:
    writer = csv.writer(file)
    writer.writerows(umap_embedding)

print('done!')


