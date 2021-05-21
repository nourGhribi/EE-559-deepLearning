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
    """
    The sigmoid function
    :param x: input
    :return: sigmoid(X)
    """
    s = 1 / (1 + x.mul(-1).exp())
    return s


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
