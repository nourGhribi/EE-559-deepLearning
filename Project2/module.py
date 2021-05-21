import torch
from torch import empty, zeros
import math


class Module(object):
    """
        Module class - all other models architecture classes in the framework should inherit from
    """
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *upstream_derivative):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    """ Fully connected layer
        Y = X * W + b.
        Parameters : dim_in and dim_out
    """
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self):
        return

    def backward(self):
        return

    def param(self):
        return []


class Sequential(Module):
    """
    Sequential
    """
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        # Take the reverse of the list of layers,
        # and do a backward pass for each layer
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def param(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.param()
        return parameters
