import modules as nn
from initializers import Xavier
from losses import LossMSE as MSE
from activations import *
from utils import *

import torch
torch.set_grad_enabled(False)

def main():
    # Model layers params 
    INPUT_UNITS = 2
    HIDDEN_LAYERS = 3
    HIDDEN_UNITS = 25
    OUTPUT_UNITS = 2
    
    lr=0.01
    
    X_train, y_train = get_data(1000)
    X_test, y_test = get_data(1000)
    
    # The model
    model = nn.Sequential(nn.Linear(INPUT_UNITS, HIDDEN_UNITS),
                          LeakyReLU(),
                          nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
                          LeakyReLU(),
                          nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
                          LeakyReLU(),
                          nn.Linear(HIDDEN_UNITS, OUTPUT_UNITS),
                          Sigmoid())
    # Train the model
    print("training the model")
    trained_model, train_loss, train_acc, test_loss, test_acc = train_test_model(model,X_train,y_train,X_test,y_test,lr=lr)  
    
if __name__ == '__main__':
    main()