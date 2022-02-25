import numpy as np
import random

from sklearn.metrics import roc_auc_score, average_precision_score



def cal_metrics(true_Y, pred_Y, y_type=None):
    '''
        - true_Y: [N X 2] array; one-hot encoding for binary label
        - pred_Y: [N x 2] array: softmax output
    '''
    
    y_dim = np.shape(true_Y)[1]
    
    if y_type == None:
        if y_dim == 1: #regression
            y_type = 'continuous'
        elif y_dim == 2: #binary classification
            y_type = 'binary'
        else: # classification - multiple categories
            y_type = 'categorical'

    if y_type == 'continuous': 
        return np.mean((true_Y.reshape([-1]) - pred_Y.reshape([-1]))**2), np.sqrt(np.mean((true_Y.reshape([-1]) - pred_Y.reshape([-1]))**2))
    elif y_type == 'binary': 
        return roc_auc_score(true_Y[:, 1], pred_Y[:, 1]), average_precision_score(true_Y[:, 1], pred_Y[:, 1])
    else: 
        K = min(5, y_dim)        
        acc = (np.argmax(true_Y, axis=1) == np.argmax(pred_Y, axis=1)[:, 0])
        for k in range(2):
            if k == 0:
                acc_k = np.copy(acc)
            else:
                acc_k |= (np.argmax(true_Y, axis=1) == np.argsort(-pred_Y, axis=1)[:, k])
        return np.mean(acc), np.mean(acc_k)
    

def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)    
    y_mb   = y[idx].astype(float)    

    return x_mb, y_mb

