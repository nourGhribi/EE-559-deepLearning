import math
import torch
from losses import LossMSE as MSE

def get_data(n = 1000):
    """
    Returns train and test data x, y each having n points sampled uniformly in [0, 1]^2,
    each with a label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 if inside.
    :param n: Number of data points
    :return: data, labels
    """
    x = torch.empty(n, 2)
    x = x.uniform_(0, 1)

    x_centered = x - 0.5
    norm_squared = x_centered.pow(2).sum(dim=1)

    radius_sq = 1 / (2 * math.pi)

    # To check if the points are inside the disk
    y = norm_squared.sub(radius_sq).sign().add(1).div(2)
    return x, y

def encode_labels(target): 
    encoded = torch.empty(target.shape[0], 2)
    for i in range(target.shape[0]): 
        if(target[i]):
            encoded[i,0] = 0
            encoded[i,1] = target[i]
        else : 
            encoded[i,0] = 1
            encoded[i,1] = target[i]
    return encoded

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=50, loss="MSE", lr="0.05", verbose=True):