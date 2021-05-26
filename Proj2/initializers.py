import torch
import math

from initializer import Initializer

class Xavier(Initializer):
    """
    Weights initializer using xavier and default unifrom distibution (else uses normal distribution)
    src=https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    """
    def __init__(self, fan_in, fan_out, dist = 'uniform'):
        super(Xavier, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.dist = dist 
    
    def initialize(self):
        
        if(self.dist == 'uniform'):
            a = math.sqrt(6. / (self.fan_in + self.fan_out) )    
            w = torch.empty(self.fan_out, self.fan_in).uniform_(-a,a)

        # normal distribution
        else: 
            std = math.sqrt(2. / (self.fan_in + self.fan_out) )
            w = tensor.normal_(0,std)
        
        return w