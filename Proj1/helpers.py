import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch import nn

import matplotlib.pyplot as plt

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

  nb_data_errors = 0

  for b in range(0, data_input.size(0), mini_batch_size):
    output,_,_ = model(data_input.narrow(0, b, mini_batch_size))      
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

def plot_model_output(trial_train_accuracies, trial_test_accuracies, trial_losses, model_name="Model"):
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