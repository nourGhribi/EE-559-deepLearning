"""
This file contains helper methods used by the three models for computing 
performance or plotting results.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch import nn

import matplotlib.pyplot as plt

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
  """
  This method computes the number of classification errors between 
  predicted data and real data.

  Parameters
  ----------
  model : torch.nn.Model
      Classification model.
  data_input : torch.tensor
      Classification input.
  data_target : torch.tensor
      Real classification output.
  mini_batch_size : int
      Size of the batches.

  Returns
  -------
  nb_data_errors : int
      Number of classification errors.

  """

  nb_data_errors = 0

  for b in range(0, data_input.size(0), mini_batch_size):
    output,_,_ = model(data_input.narrow(0, b, mini_batch_size))      
    _, predicted_classes = torch.max(output, 1)
    for k in range(mini_batch_size):
      if data_target[b + k] != predicted_classes[k]:
        nb_data_errors = nb_data_errors + 1

  return nb_data_errors

def train_report(model, train_input, train_target, test_input, test_target, mini_batch_size):
  """
  This method computes train and test accuracies and errors during a single 
  epoch.

  Parameters
  ----------
  model : torch.nn.Model
    Classification model.
  train_input : torch.tensor
    Train classification input.
  train_target : torch.tensor
    Train classification target.
  test_input : torch.tensor
    Test classification input.
  test_target : torch.tensor
    Test classification target.
  mini_batch_size : int
      Size of the batches.

  Returns
  -------
  train_accuracy : int
    Train accuracy.
  test_accuracy : int
    Test accuracy.
  train_error : int
    Train error.
  test_error : int
    Test error.

  """
  
  train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100 
  test_error  = compute_nb_errors(model, test_input,  test_target,  mini_batch_size) / test_input.size(0)  * 100 
  train_accuracy = 100 - train_error
  test_accuracy  = 100 - test_error
  return train_accuracy, test_accuracy, train_error, test_error

def plot_model_output(trial_train_accuracies, trial_test_accuracies, trial_losses, model_name="Model"):
  """
  This method plots the train and test accuracies of a certain model as well as
  its respective losses.

  Parameters
  ----------
  trial_train_accuracies : torch.tensor[nb_trials,nb_epoch]
    2d tensor with train accuracies.
  trial_test_accuracies : torch.tensor[nb_trials,nb_epoch]
    2d tensor with test accuracies.
  trial_losses : torch.tensor[nb_trials,nb_epoch]
    2d tensor with losses.
  model_name : string, optional
    Nodel name, for plot title. The default is "Model".

  Returns
  -------
  None.

  """
  nb_trials, nb_epoch = trial_train_accuracies.size()
  trial_train_accuracies_std, trial_train_accuracies_mean = torch.std_mean(trial_train_accuracies, axis = 0)
  trial_test_accuracies_std, trial_test_accuracies_mean = torch.std_mean(trial_test_accuracies, axis = 0)
  trial_losses_std, trial_losses_mean = torch.std_mean(trial_losses, axis = 0)
  
  fig, axes = plt.subplots(1, 2, figsize=(12, 3))
  ## Accuracies
  axes[0].plot(range(1,nb_epoch +1),trial_train_accuracies_mean, label="train")
  axes[0].plot(range(1,nb_epoch +1),trial_test_accuracies_mean, label="test")
  axes[0].fill_between(x= range(1,nb_epoch +1),y1= trial_train_accuracies_mean + trial_train_accuracies_std, y2= trial_train_accuracies_mean - trial_train_accuracies_std, alpha = 0.5)
  axes[0].fill_between(x= range(1,nb_epoch +1), y1= trial_test_accuracies_mean + trial_test_accuracies_std, y2= trial_test_accuracies_mean - trial_test_accuracies_std, alpha = 0.5)
  axes[0].set_ylim(50,110)
  axes[0].set_xlim(1,nb_epoch)
  axes[0].set_xlabel("epoch")
  axes[0].set_ylabel("accuracy [%]")
  axes[0].set_title("{0} accuracy for train and test datasets, over {1} trials.".format(model_name,nb_trials))
  axes[0].grid()
  axes[0].legend()
  ## Losses
  axes[1].plot(range(1,nb_epoch +1),trial_losses_mean)
  axes[1].fill_between(x= range(1,nb_epoch +1),y1= trial_losses_mean + trial_losses_std, y2= trial_losses_mean - trial_losses_std, alpha = 0.5)
  axes[1].set_xlim(1,nb_epoch)
  axes[1].set_xlabel("epoch")
  axes[1].set_ylabel("Loss")
  axes[1].set_title("{0} loss, over {1} trials.".format(model_name,nb_trials))
  axes[1].grid()
  axes[1].legend()



def plot_hyperparameters_output(trial_train_accuracies_list, trial_test_accuracies_list, trial_losses_list, model_name, hyperparameter_name, hyperparameter_values):
  """
  This method plots the train and test accuracies of a certain model as well as
  its respective losses with respect to a list of values of a certain hyperparameter.

  Parameters
  ----------
  trial_train_accuracies_list : list of torch.tensor[nb_trials,nb_epoch]
    list of 2d tensors with train accuracies.
  trial_test_accuracies_list : list of torch.tensor[nb_trials,nb_epoch]
    list 2d tensors with test accuracies.
  trial_losses_list : torch.tensor[nb_trials,nb_epoch]
    list of 2d tensors with losses.
  model_name : string
    Nodel name, for plot title.
  hyperparameter_name : string
    Name of the hyperparameter.
  hyperparameter_name : list
    List of values of the tested hyperparameter.
  Returns
  -------
  None.

  """
  assert len(trial_train_accuracies_list) == len(trial_test_accuracies_list)
  assert len(trial_train_accuracies_list) == len(trial_losses_list)
  assert len(trial_train_accuracies_list) == len(hyperparameter_values)

  fig, axes = plt.subplots(1, 3, figsize=(18, 3))

  for trial_train_accuracies, trial_test_accuracies, trial_losses, param_value in zip(trial_train_accuracies_list, trial_test_accuracies_list, trial_losses_list, hyperparameter_values):

    nb_trials, nb_epoch = trial_train_accuracies.size()
    trial_train_accuracies_std, trial_train_accuracies_mean = torch.std_mean(trial_train_accuracies, axis = 0)
    trial_test_accuracies_std, trial_test_accuracies_mean = torch.std_mean(trial_test_accuracies, axis = 0)
    trial_losses_std, trial_losses_mean = torch.std_mean(trial_losses, axis = 0)
    
  
    ## Accuracies
    axes[0].plot(range(1,nb_epoch +1),trial_train_accuracies_mean, label="train, val = {}".format(param_value))
    axes[1].plot(range(1,nb_epoch +1),trial_test_accuracies_mean, label="test, val = {}".format(param_value))
    axes[0].fill_between(x= range(1,nb_epoch +1),y1= trial_train_accuracies_mean + trial_train_accuracies_std, y2= trial_train_accuracies_mean - trial_train_accuracies_std, alpha = 0.5)
    axes[1].fill_between(x= range(1,nb_epoch +1), y1= trial_test_accuracies_mean + trial_test_accuracies_std, y2= trial_test_accuracies_mean - trial_test_accuracies_std, alpha = 0.5)
    
    ## Losses
    axes[2].plot(range(1,nb_epoch +1),trial_losses_mean, label="loss, val = {}".format(param_value))
    axes[2].fill_between(x= range(1,nb_epoch +1),y1= trial_losses_mean + trial_losses_std, y2= trial_losses_mean - trial_losses_std, alpha = 0.5)
    


  axes[0].set_ylim(40,110)
  axes[0].set_xlim(1,nb_epoch)
  axes[0].set_xlabel("epoch")
  axes[0].set_ylabel("accuracy [%]")
  axes[0].set_title("{0} accuracy for train dataset, \nover {1} trials for different {2} values.".format(model_name,nb_trials,hyperparameter_name))
  axes[0].grid()
  axes[0].legend()

  axes[1].set_ylim(40,110)
  axes[1].set_xlim(1,nb_epoch)
  axes[1].set_xlabel("epoch")
  axes[1].set_ylabel("accuracy [%]")
  axes[1].set_title("{0} accuracy for test dataset, \nover {1} trials for different {2} values.".format(model_name,nb_trials,hyperparameter_name))
  axes[1].grid()
  axes[1].legend()

  axes[2].set_xlim(1,nb_epoch)
  axes[2].set_xlabel("epoch")
  axes[2].set_ylabel("Loss")
  axes[2].set_title("{0} loss, \nover {1} trials for different {2} values.".format(model_name,nb_trials,hyperparameter_name))
  axes[2].grid()
  axes[2].legend()


def compare_model_output(trial_train_accuracies_list, trial_test_accuracies_list, trial_losses_list, model_names):
  """
  This method plots the train and test accuracies of a list of models as well as
  their respective losses.

  Parameters
  ----------
  trial_train_accuracies_list : list of torch.tensor[nb_trials,nb_epoch]
    list of 2d tensors with train accuracies.
  trial_test_accuracies_list : list of torch.tensor[nb_trials,nb_epoch]
    list 2d tensors with test accuracies.
  trial_losses_list : torch.tensor[nb_trials,nb_epoch]
    list of 2d tensors with losses.
  model_names : string
    List of model names, for plot title.
  Returns
  -------
  None.

  """
  assert len(trial_train_accuracies_list) == len(trial_test_accuracies_list)
  assert len(trial_train_accuracies_list) == len(trial_losses_list)
  assert len(trial_train_accuracies_list) == len(model_names)

  fig, axes = plt.subplots(1, 3, figsize=(18, 3))

  for trial_train_accuracies, trial_test_accuracies, trial_losses, model_name in zip(trial_train_accuracies_list, trial_test_accuracies_list, trial_losses_list, model_names):

    nb_trials, nb_epoch = trial_train_accuracies.size()
    trial_train_accuracies_std, trial_train_accuracies_mean = torch.std_mean(trial_train_accuracies, axis = 0)
    trial_test_accuracies_std, trial_test_accuracies_mean = torch.std_mean(trial_test_accuracies, axis = 0)
    trial_losses_std, trial_losses_mean = torch.std_mean(trial_losses, axis = 0)
    
  
    ## Accuracies
    axes[0].plot(range(1,nb_epoch +1),trial_train_accuracies_mean, label="train, {}".format(model_name))
    axes[1].plot(range(1,nb_epoch +1),trial_test_accuracies_mean, label="test, {}".format(model_name))
    axes[0].fill_between(x= range(1,nb_epoch +1),y1= trial_train_accuracies_mean + trial_train_accuracies_std, y2= trial_train_accuracies_mean - trial_train_accuracies_std, alpha = 0.5)
    axes[1].fill_between(x= range(1,nb_epoch +1), y1= trial_test_accuracies_mean + trial_test_accuracies_std, y2= trial_test_accuracies_mean - trial_test_accuracies_std, alpha = 0.5)
    
    ## Losses
    axes[2].plot(range(1,nb_epoch +1),trial_losses_mean, label="loss, {}".format(model_name))
    axes[2].fill_between(x= range(1,nb_epoch +1),y1= trial_losses_mean + trial_losses_std, y2= trial_losses_mean - trial_losses_std, alpha = 0.5)
    


  axes[0].set_ylim(50,110)
  axes[0].set_xlim(1,nb_epoch)
  axes[0].set_xlabel("epoch")
  axes[0].set_ylabel("accuracy [%]")
  axes[0].set_title("Aaccuracy for train datasets, \nover {1} trials for different models.".format(model_name,nb_trials))
  axes[0].grid()
  axes[0].legend()

  axes[1].set_ylim(50,110)
  axes[1].set_xlim(1,nb_epoch)
  axes[1].set_xlabel("epoch")
  axes[1].set_ylabel("accuracy [%]")
  axes[1].set_title("Accuracy for test datasets, \nover {1} trials for different models.".format(model_name,nb_trials))
  axes[1].grid()
  axes[1].legend()

  axes[2].set_xlim(1,nb_epoch)
  axes[2].set_xlabel("epoch")
  axes[2].set_ylabel("Loss")
  axes[2].set_title("Losses, over {1} trials \nfor different models.".format(model_name,nb_trials))
  axes[2].grid()
  axes[2].legend()