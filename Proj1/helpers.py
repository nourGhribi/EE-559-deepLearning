import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch import nn

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

  nb_data_errors = 0

  for b in range(0, data_input.size(0), mini_batch_size):
    try:
      output,_,_ = model(data_input.narrow(0, b, mini_batch_size))
    except:
      output = model(data_input.narrow(0, b, mini_batch_size))
    _, predicted_classes = torch.max(output, 1)
    for k in range(mini_batch_size):
      if data_target[b + k] != predicted_classes[k]:
        nb_data_errors = nb_data_errors + 1

  return nb_data_errors

def train_report(model, train_input, train_target, test_input, test_target, mini_batch_size):
  
  train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100 
  test_error  = compute_nb_errors(model, test_input,  test_target,  mini_batch_size) / test_input.size(0)  * 100 
  train_accuracy = 100 - train_error
  test_accuracy  = 100 - test_error
  return train_accuracy, test_accuracy, train_error, test_error
