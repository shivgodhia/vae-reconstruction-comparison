from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import metrics
from vae import VAE
from conditional_vae import CVAE
import utils as u

# initialise some variables
num_epochs = 10
batch_size = 1000


# Load Data 
stud_data = pd.read_csv('stud_data.csv')

X, cond, sub_df = u.build_training_set(stud_data, 300)

X_train, X_test, cond_train, cond_test = train_test_split(X, cond,test_size=0.2)

# NOTE: We are using cvae-kl and vae-mmd, the rest are for testing

# Instantiate and train a cvae with kl loss
cvae_kl = CVAE(X.shape[1], cond.shape[1], 2, [64, 32], drop_out=0, loss='kl')
cvae_kl._model.fit(x=[X_train, cond_train], y=X_train, validation_data=([X_test, cond_test], X_test), epochs=num_epochs, batch_size=batch_size)


# Instantiate and train a vae with mmd loss
vae_mmd = VAE(X.shape[1], 2, [4], drop_out=0.2, loss='mmd')
vae_mmd.fit(X_train, X_test, batch_size=batch_size, epochs=num_epochs)


# Instantiate and train a cvae with mmd loss
cvae_mmd = CVAE(X.shape[1], cond.shape[1], 2, [64, 32], drop_out=0, loss='mmd')
cvae_mmd._model.fit(x=[X_train, cond_train], y=X_train, validation_data=(
    [X_test, cond_test], X_test), epochs=num_epochs, batch_size=batch_size)


# Instanciate and train a vae with kl loss
vae_kl = VAE(X.shape[1], 2, [4], drop_out=0.2, loss='kl')
vae_kl.fit(X_train, X_test, batch_size=batch_size, epochs=num_epochs)


# Plot latent spaces for both models
u.plot_latent_space([X, cond], cvae_kl,[sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'CVAE_kl',s=1)
u.plot_latent_space(X, vae_mmd,
                    [sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'VAE_mmd',
                    s=1) 

# plot training history (loss over time) and metric (metric over time)


u.plot_training_history([vae_kl, vae_mmd, cvae_mmd, cvae_kl], 'mean_squared_error', ['VAE-kl', 'VAE-mmd', 'CVAE-mmd', 'CVAE-kl'], "Comparison of Reconstruction Loss (MSE) between VAEs and CVAEs")

u.plot_training_history([vae_kl, vae_mmd, cvae_mmd, cvae_kl], 'mean_absolute_error', ['VAE-kl', 'VAE-mmd', 'CVAE-mmd', 'CVAE-kl' ], "Comparison of Reconstruction Loss (MAE) between VAEs and CVAEs")


# compare the reconstruction loss between the VAE and the CVAE. 
# My assumption is that the CVAE allows a better reconstruction of the original parameters.
# Here is a suggested list of sub-tasks:
# -  Read some Documentation on how  metrics work with keras (how to implement them, how to use some existing layers to build a custom metric).
# -  Identify the relevant layers in the current architectures. (We only want to compare the input numerical parameters and the reconstructed output, not the one hot encoded vectors )
# -  Implement the reconstruction metric.
# -  Compare the reconstruction score between vae and cvae
