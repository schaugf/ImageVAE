# VAE analysis

library(optparse)
library(ggplot2)
library(tidyr)
library(RColorBrewer)
library(Rtsne)

option_list = list(
  make_option(c('-d', '--results_dir'), type='character', default=NA,
              help='VAE results dir'),
  make_option(c('-s', '--save_dir'), type='character', default='plots',
              help='save directory'),
  make_option(c('-x', '--img_width'), type='numeric', default=5,
              help='image width (in)'),
  make_option(c('-y', '--img_height'), type='numeric', default=4,
              help='image height (in)')
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.na(opt$results_dir)) { stop('encoding file missing') }

opt$save_dir = file.path(opt$results_dir, opt$save_dir)

dir.create(opt$save_dir, showWarnings=FALSE)


#	load data files
cat('loading data files...\n')

encodings <- data.frame(read.csv(file.path(opt$results_dir, 'encodings.csv'), header=F))
names(encodings) <- paste0('vae', seq(1:ncol(encodings)))

umap_embedding = data.frame(read.csv(file.path(opt$results_dir, 'umap_embedding.csv'), header=F))
names(umap_embedding) = c('umap1', 'umap2')

tsne_embedding = data.frame(read.csv(file.path(opt$results_dir, 'tsne_embedding.csv'), header=F))
names(tsne_embedding) = c('tsne1', 'tsne2')

zvar = data.frame(read.csv(file.path(opt$results_dir, 'z_log_var.csv'), header=F))
names(zvar) <- paste0('zvar', seq(1:ncol(encodings)))

zmean = data.frame(read.csv(file.path(opt$results_dir, 'z_mean.csv'), header=F))
names(zmean) <- paste0('zmean', seq(1:ncol(encodings)))

master_df <- cbind(encodings, umap_embedding, tsne_embedding)


#	load training log

training_log = data.frame(read.csv(file.path(opt$results_dir, 'training.log'), header=T, sep='\t'))

#	color palette
myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-3, 3))
sf <- scale_fill_gradientn(colours = myPalette(100), limits=c(-3, 3))

#	plot loss trends

cat('generating plots...\n')

ggplot(training_log) +
		geom_point(aes(x=epoch, y=log(loss))) +
    geom_line(aes(x=epoch, y=log(loss))) +
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

# plot encoding densities

pd = encodings %>%
  gather()

ggplot(pd) +
  geom_density(aes(x=value, fill=key), alpha=0.2) +
  theme_minimal() +
  ggsave(file.path(opt$save_dir, 'latent_distribution.pdf'),
         height=opt$img_height, width=opt$img_width, units='in', device='pdf')

cat('done! \n')





