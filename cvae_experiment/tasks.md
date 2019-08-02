# Tasks

I have cleaned the code and compress the project into a single tar file. I advise you to start from this version of the project. I have added some comments and functions.

Your goal for next Tuesday, is to

**compare the reconstruction loss between the VAE and the CVAE.**

My assumption is that the CVAE allows a better reconstruction of the original parameters.

## Sub-tasks

Here is a suggested list of sub-tasks:

* ~~Read some Documentation on how  metrics work with keras (how to implement them, how to use some existing layers to build a custom metric).~~
* Identify the relevant layers in the current architectures. (We only want to compare the input numerical parameters and the reconstructed output, not the one hot encoded vectors )
* Implement the reconstruction metric.
* Compare the reconstruction score between vae and cvae.



## Slides for iX Presentation

* can explain in our own words what a vae is, what a cvae is, etc
* ran experiment for DP and want to explain whether we should or should not replace current VAE for the client (with a CVAE)
* no metric has really been found to fully evaluate the quality of the latent space (metric should be correlated w quality)
* what we are doing now is to plot random latent spaces from random weld locations
* CVAE may not improve the separation but it improves the MSE
* representation in the latent space is better in quality using CVAE
