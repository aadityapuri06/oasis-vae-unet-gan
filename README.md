# OASIS Brain MRI — VAE, UNet, GAN

Three deep learning models trained on the OASIS brain MRI dataset (2D slices),
for COMP3710 Pattern Recognition, Part 4.

## [vae/](vae/) — Variational Autoencoder
Encoder/decoder trained on brain MRI slices, with the reparameterization
trick and a KL-divergence-regularized latent space. Includes reconstruction
comparisons, a UMAP projection of the learned latent space, and a smooth
latent-space interpolation walk demonstrating the manifold's continuity.

## [unet/](unet/) — UNet Segmentation
Brain tissue segmentation (Background, CSF, Gray Matter, White Matter) using
inverse-frequency class-weighted loss to handle class imbalance. Achieves
Dice scores above 0.9 on every class on the held-out test set (Background
0.998, CSF 0.939, Gray Matter 0.947, White Matter 0.971). Trained on
Rangpur (A100 GPU).

## [gan/](gan/) — Generative Adversarial Network
DCGAN-style Generator/Discriminator trained on the OASIS dataset to generate
novel, realistic brain MRI slices. Uses one-sided label smoothing for
training stability. Trained for 50 epochs on Rangpur; loss plot and sample
image grids from multiple epochs are included as evidence of training. No
mode collapse observed — see gan/gan_samples_epoch_*.png for progression.