#!/usr/bin/bash

if [ $# -eq 0 ]
then
		echo "No arguments supplied"
		exit 1
fi

echo 'running ImageVAE analysis on' $1

source activate DL

#	generate umap projection

python src/umap_embed.py \
		$1/encodings.csv \
		$1/umap_embedding.csv	
		
#	generate tsne plots and loss trends

Rscript src/vae_analysis.R \
		-e $1/encodings.csv \
		-u $1/umap_embedding.csv \
		-l $1/training.log \
		-s $1/analysis \

mv $1/umap_embedding.csv $1/analysis


