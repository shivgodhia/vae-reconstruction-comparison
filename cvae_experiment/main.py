from sklearn.model_selection import train_test_split
import pandas as pd
from keras import metrics
import matplotlib as plot
from vae import VAE
from conditional_vae import CVAE
import utils as u

# Load Data 
stud_data = pd.read_csv('stud_data.csv')

X, cond, sub_df = u.build_training_set(stud_data, 300)

X_train, X_test, cond_train, cond_test = train_test_split(X, cond,
                                                          test_size=0.2)

# Instanciate and train a cvae with kl loss
cvae = CVAE(X.shape[1], cond.shape[1], 2, [64, 32], drop_out=0, loss='kl')
cvae._model.fit(x=[X_train, cond_train], y=X_train,
                validation_data=([X_test, cond_test], X_test),
                epochs=3, batch_size=1000)


# Instanciate and train a vae with mmd losst
vae = VAE(X.shape[1], 2, [4], drop_out=0.2, loss='mmd')
vae.fit(X_train, X_test, batch_size=000, epochs=3)


# Plot latent spaces for both models
u.plot_latent_space([X, cond], cvae,
                    [sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'CVAE_kl',
                    s=1)
u.plot_latent_space(X, vae,
                    [sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'VAE_mmd',
                    s=1) 


plt.plot(vae.history.history['mse'])

# compare the reconstruction loss between the VAE and the CVAE. 
# My assumption is that the CVAE allows a better reconstruction of the original parameters.
# Here is a suggested list of sub-tasks:
# -  Read some Documentation on how  metrics work with keras (how to implement them, how to use some existing layers to build a custom metric).
# -  Identify the relevant layers in the current architectures. (We only want to compare the input numerical parameters and the reconstructed output, not the one hot encoded vectors )
# -  Implement the reconstruction metric.
# -  Compare the reconstruction score between vae and cvae