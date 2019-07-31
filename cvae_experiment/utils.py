from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


controllable_params = ['Lift Height act', 'WeldTime Act',  'Stickout', 'WeldCurrent act']
noncontrollable_params = ['DropTime act', 'Pilot WeldCurrent Arc Voltage Act', 'Weld Energy act']
features = controllable_params + noncontrollable_params


def get_most_populated_weld_location(list_id, n):
    count = list_id.value_counts()
    return list(count.iloc[:n].index)


def build_training_set(stud_data, n):
    '''
    Extract the relevant features from the top n most populated weld locations.
    Extracted features:
        X: numpy array with scaled numerical features
        cond: One hot encoded vectors for the weld location ID
        sub_df[['StudID', 'quality', 'Penetration act']]: Pandas Data Frame with columns 
                                                          that can be used as overlay in the latent space.
    '''
    list_wl = get_most_populated_weld_location(stud_data['StudID'], n)
    sub_df = stud_data.loc[stud_data['StudID'].isin(list_wl)]
    X = sub_df[features]
    X = MinMaxScaler().fit_transform(X)
    cond = pd.get_dummies(sub_df['StudID']).values
    sub_df['StudID'] = LabelEncoder().fit_transform(sub_df['StudID'])
    return X, cond, sub_df[['StudID', 'quality', 'Penetration act']]


def plot_latent_space(X, vae, overlays, title, n_max=200000, **kwarg):
    len_X = len(X[0]) if type(X) == list else len(X)
    ind = np.arange(len_X)
    if len_X > n_max:
        ind = np.random.choice(ind, n_max, replace=False)
    if type(X) == list:
        X = [x[ind] for x in X]
    else:
        X = X[ind]
    ls = vae.encode(X)
    n_cols = 2
    n_rows = int(len(overlays) / 2 + 1)
    fig = plt.figure(figsize=(15, 10))
    for i, overlay in enumerate(overlays):
        plt.subplot(n_rows, n_cols, i+1)
        overlay = overlay.iloc[ind]
        if overlay.name == 'Penetration act':
            overlay = overlay.clip(0.3, 1.7)
        plt.scatter(ls[:, 0],
                    ls[:, 1],
                    c=overlay,
                    **kwarg)
        plt.colorbar()
        plt.title(overlay.name)
    fig.suptitle(title)


def plot_weld_location_ls(i, list_id, X, vae, overlay, **kwarg):
    lis_id_unique = np.unique(list_id)
    mask = list_id == lis_id_unique[i]
    if type(X) == list:
        X = [x[mask] for x in X]
    else:
        X = X[mask]
    ls = vae.encode(X)
    overlay = overlay.loc[mask]
    if overlay.name == 'Penetration act':
        overlay = overlay.clip(0.3, 1.7)
    elif overlay.name == 'quality':
        overlay = overlay.clip(0.4, 1)
    overlay = (overlay - 0.4) / 0.6
    plt.figure(figsize=(13, 10))
    plt.scatter(ls[:, 0],
                ls[:, 1],
                c=overlay,
                **kwarg)
    plt.colorbar()


def plot_training_history(keras_models, metric, labels):
    '''
    keras_models: list of keras models
    metric: the name of the metric to plot
    labels: list of model names
    '''
    for model, label in zip(keras_models, labels):
