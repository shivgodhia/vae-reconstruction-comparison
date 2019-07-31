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
