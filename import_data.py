import pandas as pd
import numpy as np

import random
import pickle
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons


def get_noisy_two_moons(n_samples=1000, n_feats=100, noise_twomoon=0.1, noise_nuisance=1.0, seed_=1234):
    X, Y = make_moons(n_samples=n_samples, noise=noise_twomoon, random_state=seed_)
    np.random.seed(seed_)
    N = np.random.normal(loc=0., scale=noise_nuisance, size=[n_samples, n_feats-2])
    X = np.concatenate([X, N], axis=1)

    Y_onehot = np.zeros([n_samples, 2])
    Y_onehot[Y == 0, 0] = 1
    Y_onehot[Y == 1, 1] = 1

    return X, Y, Y_onehot

def get_blockcorr(X, block_size=10, noise_=0.5, seed_=1234): 
    '''
        noise 0.5 ~ 0.85 correlation
        noise 1.0 ~ 0.66 correlation
    '''
    for p in range(X.shape[1]):
        np.random.seed(seed_ + p)
        tmp   = X[:, [p]] + np.random.normal(loc=0., scale=noise_, size=[X.shape[0], block_size-1])

        if p == 0:
            X_new = np.concatenate([X[:, [p]], tmp], axis=1)
        else:
            X_new = np.concatenate([X_new, X[:, [p]], tmp], axis=1)    
    return X_new   