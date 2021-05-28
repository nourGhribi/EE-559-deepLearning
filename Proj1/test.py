import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch import nn
from torch import cuda

import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import dlc_practical_prologue as prologue

from models import *
from helpers import *

nb_samples = 1000
print("Generating random dataset of {} samples...".format(nb_samples))
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb_samples)

# work on GPU if device is available
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device " + device_name)
train_input, train_target, train_classes, test_input, test_target, test_classes = train_input.to(device), train_target.to(device), train_classes.to(device), test_input.to(device), test_target.to(device), test_classes.to(device)

print('train_input:', train_input.size(), 'train_classes:', train_classes.size(), 'train_target:', train_target.size())
print('test_input:', test_input.size(), 'test_classes:', test_classes.size(), 'test_target:', test_target.size())


#DeepModel auxiliary_loss_weight_deep = 0
optimizer_deep = optim.Adam
nb_epoch_deep = 25
lr_deep = 0.005
mini_batch_size_deep = 100
nb_hidden_deep = 100
auxiliary_loss_weight_deep = 0
print("Loading pickled DeepModel...")
deep = DeepModel(nb_hidden_deep, nb_epoch_deep, lr_deep, mini_batch_size_deep, auxiliary_loss_weight_deep).to(device)
deep.load_state_dict(torch.load('pickles/deep_00/best_model.pt'))
print("Training DeepModel...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(deep, train_input, train_target, test_input, test_target, mini_batch_size_deep)
print("Obtained test accuracy for DeepModel is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}, auxiliary_loss_weight = {5}".format(optimizer_deep, nb_epoch_deep, lr_deep, mini_batch_size_deep, nb_hidden_deep, auxiliary_loss_weight_deep))


#DeepModel auxiliary_loss_weight_deep = 0.5
optimizer_deep = optim.Adam
nb_epoch_deep = 25
lr_deep = 0.005
mini_batch_size_deep = 100
nb_hidden_deep = 100
auxiliary_loss_weight_deep = 0.5
print("Loading pickled DeepModel...")
deep = DeepModel(nb_hidden_deep, nb_epoch_deep, lr_deep, mini_batch_size_deep, auxiliary_loss_weight_deep).to(device)
deep.load_state_dict(torch.load('pickles/deep_05/best_model.pt'))
print("Training DeepModel...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(deep, train_input, train_target, test_input, test_target, mini_batch_size_deep)
print("Obtained test accuracy for DeepModel is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}, auxiliary_loss_weight = {5}".format(optimizer_deep, nb_epoch_deep, lr_deep, mini_batch_size_deep, nb_hidden_deep, auxiliary_loss_weight_deep))


#DeepModel auxiliary_loss_weight_deep = 1.0
optimizer_deep = optim.Adam
nb_epoch_deep = 25
lr_deep = 0.005
mini_batch_size_deep = 100
nb_hidden_deep = 100
auxiliary_loss_weight_deep = 1.0
print("Loading pickled DeepModel...")
deep = DeepModel(nb_hidden_deep, nb_epoch_deep, lr_deep, mini_batch_size_deep, auxiliary_loss_weight_deep).to(device)
deep.load_state_dict(torch.load('pickles/deep_10/best_model.pt'))
print("Training DeepModel...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(deep, train_input, train_target, test_input, test_target, mini_batch_size_deep)
print("Obtained test accuracy for DeepModel is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}, auxiliary_loss_weight = {5}".format(optimizer_deep, nb_epoch_deep, lr_deep, mini_batch_size_deep, nb_hidden_deep, auxiliary_loss_weight_deep))


#DeepModelWS auxiliary_loss_weight_deep_ws = 0
optimizer_deep_ws = optim.Adam
nb_epoch_deep_ws = 25
lr_deep_ws = 0.005
mini_batch_size_deep_ws = 100
nb_hidden_deep_ws = 500
auxiliary_loss_weight_deep_ws = 0
print("Loading pickled DeepModelWS...")
deep_ws = DeepModelWS(nb_hidden_deep_ws, nb_epoch_deep_ws, lr_deep_ws, mini_batch_size_deep_ws, auxiliary_loss_weight_deep_ws).to(device)
deep_ws.load_state_dict(torch.load('pickles/deep_ws_00/best_model.pt'))
print("Training DeepModelWS...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(deep_ws, train_input, train_target, test_input, test_target, mini_batch_size_deep_ws)
print("Obtained test accuracy for DeepModelWS is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}, auxiliary_loss_weight = {5}".format(optimizer_deep_ws, nb_epoch_deep_ws, lr_deep_ws, mini_batch_size_deep_ws, nb_hidden_deep_ws, auxiliary_loss_weight_deep_ws))


#DeepModelWS auxiliary_loss_weight_deep_ws = 0.5
optimizer_deep_ws = optim.Adam
nb_epoch_deep_ws = 25
lr_deep_ws = 0.005
mini_batch_size_deep_ws = 100
nb_hidden_deep_ws = 500
auxiliary_loss_weight_deep_ws = 0.5
print("Loading pickled DeepModelWS...")
deep_ws = DeepModelWS(nb_hidden_deep_ws, nb_epoch_deep_ws, lr_deep_ws, mini_batch_size_deep_ws, auxiliary_loss_weight_deep_ws).to(device)
deep_ws.load_state_dict(torch.load('pickles/deep_ws_05/best_model.pt'))
print("Training DeepModelWS...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(deep_ws, train_input, train_target, test_input, test_target, mini_batch_size_deep_ws)
print("Obtained test accuracy for DeepModelWS is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}, auxiliary_loss_weight = {5}".format(optimizer_deep_ws, nb_epoch_deep_ws, lr_deep_ws, mini_batch_size_deep_ws, nb_hidden_deep_ws, auxiliary_loss_weight_deep_ws))


#DeepModelWS auxiliary_loss_weight_deep_ws = 1.0
optimizer_deep_ws = optim.Adam
nb_epoch_deep_ws = 25
lr_deep_ws = 0.005
mini_batch_size_deep_ws = 100
nb_hidden_deep_ws = 500
auxiliary_loss_weight_deep_ws = 1.0
print("Loading pickled DeepModelWS...")
deep_ws = DeepModelWS(nb_hidden_deep_ws, nb_epoch_deep_ws, lr_deep_ws, mini_batch_size_deep_ws, auxiliary_loss_weight_deep_ws).to(device)
deep_ws.load_state_dict(torch.load('pickles/deep_ws_10/best_model.pt'))
print("Training DeepModelWS...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(deep_ws, train_input, train_target, test_input, test_target, mini_batch_size_deep_ws)
print("Obtained test accuracy for DeepModelWS is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}, auxiliary_loss_weight = {5}".format(optimizer_deep_ws, nb_epoch_deep_ws, lr_deep_ws, mini_batch_size_deep_ws, nb_hidden_deep_ws, auxiliary_loss_weight_deep_ws))


#ShallowModel
optimizer_shallow = optim.Adam
nb_epoch_shallow = 25
lr_shallow = 0.0007
mini_batch_size_shallow = 100
nb_hidden_shallow = 500
print("Loading pickled ShallowModel...")
shallow = ShallowModel(nb_hidden_shallow, nb_epoch_shallow, lr_shallow, mini_batch_size_shallow).to(device)
shallow.load_state_dict(torch.load('pickles/shallow/best_model.pt'))
print("Training ShallowModel...")
train_accuracy, test_accuracy, train_error, test_error = helpers.train_report(shallow, train_input, train_target, test_input, test_target, mini_batch_size_shallow)
print("Obtained test accuracy for ShallowModel is {0} and train accuracy is {1}, using the following parameters:".format(test_accuracy, train_accuracy))
print("optimizer = {0}, nb_epoch = {1}, lr = {2}, mini_batch_size = {3}, nb_hidden = {4}".format(optimizer_shallow, nb_epoch_shallow, lr_shallow, mini_batch_size_shallow, nb_hidden_shallow))