from module import Module

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        y = x * (x > 0).float()
        return y

    def backward(self, dl_dz):
        dz_dx = 0.5 * (self.x.sign() + 1)
        dl_dx = dz_dx * dl_dz
        return dl_dx
    
class LeakyReLU(Module):

    def __init__(self,negative_slope=0.01):
        #negative_slope – Controls the angle of the negative slope. Default: 1e-2

        super(LeakyReLU, self).__init__()
        self.x = None
        self.negative_slope = negative_slope

    def forward(self,x):
        self.x = x
        y = x * self.negative_slope * (x <= 0).float() +  x * (x > 0).float()
        return y

    def backward(self, dl_dz):
        dz_dx =  self.negative_slope * (self.x <= 0).float() +  (self.x > 0).float()
        dl_dx = dz_dx * dl_dz
        return dl_dx



class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return x.tanh()

    def backward(self, dl_dz):
        dz_dx = 1 - self.x.tanh().pow(2)
        dl_dx = dz_dx * dl_dz
        return dl_dx


def sigmoid(x):
    return 1 / (1 + x.mul(-1).exp())


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, dl_dz):
        f_x = sigmoid(self.x)
        dz_dx = f_x * (1 - f_x)
        dl_dx = dz_dx * dl_dz
        return dl_dx