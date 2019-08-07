from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import metrics
from vae import VAE
from conditional_vae import CVAE
import utils as u

def main(num_epochs=10, batch_size=1000):
    """Instantiates, trains and plots the training history of a VAE and CVAE
    
    :param num_epochs: number of epochs to train both vaes for, defaults to 10
    :type num_epochs: int, optional
    :param batch_size: Number of samples per gradient update, defaults to 1000
    :type batch_size: int, optional
    """
    # Load Data and split it
    stud_data = pd.read_csv('stud_data.csv')

    X, cond, sub_df = u.build_training_set(stud_data, 300)

    X_train, X_test, cond_train, cond_test = train_test_split(X, cond,test_size=0.2)

    # NOTE: We are choosing cvae-kl and vae-mmd, cvae-mmd and vae-kl are just for testing purposes

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

    # Instantiate and train a vae with kl loss
    vae_kl = VAE(X.shape[1], 2, [4], drop_out=0.2, loss='kl')
    vae_kl.fit(X_train, X_test, batch_size=batch_size, epochs=num_epochs)


    # Plot latent spaces for all models
    u.plot_latent_space([X, cond], cvae_kl,[sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'CVAE_kl',s=1)
    u.plot_latent_space(X, vae_mmd,[sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'VAE_mmd',s=1)
    u.plot_latent_space([X, cond], cvae_mmd, [sub_df['StudID'],sub_df['quality'], sub_df['Penetration act']], 'CVAE_mmd', s=1)
    u.plot_latent_space(X, vae_kl, [sub_df['StudID'], sub_df['quality'], sub_df['Penetration act']], 'VAE_kd', s=1)

    # plot training history (loss over time) and metric (metric over time)
    u.plot_training_history([vae_kl, vae_mmd, cvae_mmd, cvae_kl], 'mean_squared_error', ['VAE-kl', 'VAE-mmd', 'CVAE-mmd', 'CVAE-kl'], "Comparison of Reconstruction Loss (MSE) between VAEs and CVAEs")

    u.plot_training_history([vae_kl, vae_mmd, cvae_mmd, cvae_kl], 'mean_absolute_error', ['VAE-kl', 'VAE-mmd', 'CVAE-mmd', 'CVAE-kl' ], "Comparison of Reconstruction Loss (MAE) between VAEs and CVAEs")

if __name__ == "__main__":
    main()
