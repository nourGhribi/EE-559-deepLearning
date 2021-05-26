import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch import nn
from torch import cuda

import helpers

class Flatten(nn.Module):
  def __init__(self):
        super().__init__()
  def forward(self, x):
      return x.view(x.size(0), -1)

class ShallowModel(nn.Module):
  def __init__(self, nb_hidden = 100, nb_epochs = 250, lr = 1e-1, mini_batch_size = 1, optimizer= optim.SGD):
    super().__init__()
    
    self.nb_hidden = nb_hidden
    self.nb_epochs = nb_epochs
    self.lr = lr
    self.mini_batch_size = mini_batch_size
    self.optim = optimizer

    self.fc = nn.Sequential(Flatten(),
                            nn.Linear( 14*14*2, self.nb_hidden),
                            nn.ReLU(),
                            nn.Linear(self.nb_hidden, 2)
                            )

  def forward(self, x):
    return self.fc(x), None, None

  def train_model(self, train_input, train_target, test_input, test_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = self.optim(self.parameters(), lr = self.lr)

    train_error = torch.zeros(self.nb_epochs)
    test_error = torch.zeros(self.nb_epochs)
    train_accuracy = torch.zeros(self.nb_epochs)
    test_accuracy = torch.zeros(self.nb_epochs)
    acc_losses = torch.zeros(self.nb_epochs)

    for e in range(self.nb_epochs):
      acc_loss = 0
      for b in range(0, train_input.size(0), self.mini_batch_size):
        output,_,_ = self(train_input.narrow(0, b, self.mini_batch_size))
        loss = criterion(output, train_target.narrow(0, b, self.mini_batch_size))
        self.zero_grad()
        loss.backward()
        optimizer.step()
        acc_loss += loss.item()
      acc_losses[e] = acc_loss
      train_accuracy[e], test_accuracy[e], train_error[e], test_error[e] = helpers.train_report(self, train_input, train_target, test_input, test_target, self.mini_batch_size)
    
    return acc_losses, train_accuracy, test_accuracy, train_error, test_error

class DeepModel(nn.Module):
  def __init__(self, nb_hidden=200, nb_epochs = 250, lr = 1e-1, mini_batch_size = 1, auxiliary_loss_weight=0, optimizer= optim.SGD):
    super().__init__()
    self.nb_hidden = nb_hidden
    self.nb_epochs = nb_epochs
    self.lr = lr
    self.mini_batch_size = mini_batch_size
    self.auxiliary_loss_weight = auxiliary_loss_weight
    self.optim = optimizer

    self.conv1= nn.Sequential(nn.Conv2d(1,32, kernel_size=3),
                              nn.MaxPool2d(kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(32,32, kernel_size=3),
                              nn.MaxPool2d(kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(32,64, kernel_size=2),
                              nn.ReLU(),
                              Flatten(),
                              nn.Linear( 64, self.nb_hidden),
                              nn.ReLU(),
                              nn.Linear(self.nb_hidden, 10)
                              )
    
    self.conv2= nn.Sequential(nn.Conv2d(1,32, kernel_size=3),
                              nn.MaxPool2d(kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(32,32, kernel_size=3),
                              nn.MaxPool2d(kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(32,64, kernel_size=2),
                              nn.ReLU(),
                              Flatten(),
                              nn.Linear( 64, self.nb_hidden),
                              nn.ReLU(),
                              nn.Linear(self.nb_hidden, 10)
                              )
    self.comp = nn.Sequential(nn.ReLU(), nn.Linear(20,2) )

  def forward(self, x):
    img_1, img_2 = torch.chunk(x, chunks=2, dim=1)
    img_1_pred = self.conv1(img_1)
    img_2_pred = self.conv2(img_2)
    img_labels = torch.cat((img_1_pred,img_2_pred), dim=1)
    pred = self.comp(img_labels)

    return pred, img_1_pred, img_2_pred

  def train_model(self, train_input, train_classes, train_target, test_input, test_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = self.optim(self.parameters(), lr = self.lr)

    class_1, class_2 = torch.chunk(train_classes , chunks=2, dim=1)

    train_error = torch.zeros(self.nb_epochs)
    test_error = torch.zeros(self.nb_epochs)
    train_accuracy = torch.zeros(self.nb_epochs)
    test_accuracy = torch.zeros(self.nb_epochs)
    acc_losses = torch.zeros(self.nb_epochs)

    for e in range(self.nb_epochs):
      acc_loss = 0
      for b in range(0, train_input.size(0), self.mini_batch_size):
        output, img_1_pred, img_2_pred = self(train_input.narrow(0, b, self.mini_batch_size))
        
        loss = (1- self.auxiliary_loss_weight)*criterion(output, train_target.narrow(0, b, self.mini_batch_size)) + \
            self.auxiliary_loss_weight * (criterion(img_1_pred, class_1.narrow(0, b, self.mini_batch_size)[:,0] ) + criterion(img_2_pred, class_2.narrow(0, b, self.mini_batch_size)[:,0] ))/2
        
        self.zero_grad()
        loss.backward()
        optimizer.step()
        acc_loss += loss.item()
      acc_losses[e] = acc_loss
      train_accuracy[e], test_accuracy[e], train_error[e], test_error[e] = helpers.train_report(self, train_input, train_target, test_input, test_target, self.mini_batch_size)
    
    return acc_losses, train_accuracy, test_accuracy, train_error, test_error
    
class DeepModelWS(nn.Module):
  def __init__(self, nb_hidden=200, nb_epochs = 250, lr = 1e-1, mini_batch_size = 1, auxiliary_loss_weight=0, optimizer= optim.SGD):
    super().__init__()
    self.nb_hidden = nb_hidden
    self.nb_epochs = nb_epochs
    self.lr = lr
    self.mini_batch_size = mini_batch_size
    self.auxiliary_loss_weight = auxiliary_loss_weight
    self.optim = optimizer

    self.conv1= nn.Sequential(nn.Conv2d(1,32, kernel_size=3),
                              nn.MaxPool2d(kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(32,32, kernel_size=3),
                              nn.MaxPool2d(kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(32,64, kernel_size=2),
                              nn.ReLU(),
                              Flatten(),
                              nn.Linear( 64, self.nb_hidden),
                              nn.ReLU(),
                              nn.Linear(self.nb_hidden, 10)
                              )
    self.comp = nn.Sequential(nn.ReLU(), nn.Linear(20,2) )

  def forward(self, x):
    img_1, img_2 = torch.chunk(x, chunks=2, dim=1)
    img_1_pred = self.conv1(img_1)
    img_2_pred = self.conv1(img_2)
    img_labels = torch.cat((img_1_pred,img_2_pred), dim=1)
    pred = self.comp(img_labels)

    return pred, img_1_pred, img_2_pred

  def train_model(self, train_input, train_classes, train_target, test_input, test_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = self.optim(self.parameters(), lr = self.lr)

    class_1, class_2 = torch.chunk(train_classes , chunks=2, dim=1)

    train_error = torch.zeros(self.nb_epochs)
    test_error = torch.zeros(self.nb_epochs)
    train_accuracy = torch.zeros(self.nb_epochs)
    test_accuracy = torch.zeros(self.nb_epochs)
    acc_losses = torch.zeros(self.nb_epochs)

    for e in range(self.nb_epochs):
      acc_loss = 0
      for b in range(0, train_input.size(0), self.mini_batch_size):
        output, img_1_pred, img_2_pred = self(train_input.narrow(0, b, self.mini_batch_size))
        
        loss = (1- self.auxiliary_loss_weight)*criterion(output, train_target.narrow(0, b, self.mini_batch_size)) + \
            self.auxiliary_loss_weight * (criterion(img_1_pred, class_1.narrow(0, b, self.mini_batch_size)[:,0] ) + criterion(img_2_pred, class_2.narrow(0, b, self.mini_batch_size)[:,0] ))/2
        
        self.zero_grad()
        loss.backward()
        optimizer.step()
        acc_loss += loss.item()
      acc_losses[e] = acc_loss
      train_accuracy[e], test_accuracy[e], train_error[e], test_error[e] = helpers.train_report(self, train_input, train_target, test_input, test_target, self.mini_batch_size)
    
    return acc_losses, train_accuracy, test_accuracy, train_error, test_error