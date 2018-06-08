# VAE analysis

library(optparse)
library(ggplot2)
library(RColorBrewer)
library(Rtsne)

option_list = list(
  make_option(c('-e', '--encoding'), type='character', default=NA,
              help='VAE encoding csv file'),
  make_option(c('-u', '--umap'), type='character', default=NA,
              help='umap embedding file'),
  make_option(c('-t', '--tsne'), type='character', default=NA,
              help='run tsne embedding'),
  make_option(c('-l', '--log'), type='character', default=NA,
  			  help='training log'),
  make_option(c('-s', '--save_dir'), type='character', default='save',
              help='save directory'),
  make_option(c('-x', '--img_width'), type='numeric', default=5,
              help='image width (in)'),
  make_option(c('-y', '--img_height'), type='numeric', default=4,
              help='image height (in)')
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.na(opt$encoding)) { stop('encoding file missing') }

dir.create(opt$save_dir, showWarnings=FALSE)


#	load data files
cat('loading data files...\n')

encoding <- data.frame(read.csv(opt$encoding, header=F))
names(encoding) <- paste0('vae', seq(1:ncol(encoding)))
cat('VAE encoding file found with', nrow(encoding), 'with', ncol(encoding), 'latent features \n')

umap_embedding = data.frame(read.csv(opt$umap, header=F))
names(umap_embedding) = c('umap1', 'umap2')
cat('UMAP embedding file found with', nrow(umap_embedding), 'with', ncol(umap_embedding), 'dimensions \n')

#	check files are similar
if (nrow(encoding) != nrow(umap_embedding)) {
  stop('different row size of encoding and umap')
}


#	load tsne file if provided
if (!is.na(opt$tsne)) {
  tsne_embedding = data.frame(read.csv(tsne_embedding.csv, header=F))
  names(tsne_embedding) = c('tsne1', 'tsne2')
  cat('tSNE embedding file found with', nrow(tsne_embedding), 'with', ncol(tnse_embedding), 'dimensions')
} else {
  cat('computing tsne projections... \n')
  tsne_data = encoding
  tsne_data[,1] = tsne_data[,1] + rnorm(nrow(tsne_data), sd=0.001)  # to remove duplicates
  tsne_results <- Rtsne(tsne_data, perplexity=30)
  tsne_embedding = data.frame(tsne_results$Y)
  names(tsne_embedding) = c('tsne1', 'tsne2')
  cat('saving tsne... \n')
  write.table(tsne_embedding, file.path(opt$save_dir, 'tsne_embedding.csv'), 
  			  quote=F, col.names=F, row.names=F, sep=',')
}

master_df <- cbind(encoding, umap_embedding, tsne_embedding)


#	load training log

training_log = data.frame(read.csv(opt$log, header=T, sep='\t'))
cat('training log found with', nrow(training_log), 'epochs \n')

#	color palette
myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-3, 3))
sf <- scale_fill_gradientn(colours = myPalette(100), limits=c(-3, 3))

#	plot loss trends

ggplot(training_log) +
		geom_point(aes(x=epoch, y=log(loss))) +
		theme_minimal() +
		xlab('training epoch') +
		ylab('log loss') +
		ggsave(file.path(opt$save_dir, 'training.pdf'), 
			   height=opt$img_height, width=opt$img_width, units='in', device='pdf')

#	plot umap projections

ggplot(umap_embedding) +
		geom_point(aes(x=umap1, y=umap2)) +
		theme_minimal() +
		ggsave(file.path(opt$save_dir, 'umap_projection.pdf'),
			   height=opt$img_height, width=opt$img_width, units='in', device='pdf')

#	plot tsne projection

ggplot(tsne_embedding) +
		geom_point(aes(x=tsne1, y=tsne2)) +
		theme_minimal() +
		ggsave(file.path(opt$save_dir, 'tsne_projection.pdf'),
			   height=opt$img_height, width=opt$img_width, units='in', device='pdf')

cat('done! \n')





