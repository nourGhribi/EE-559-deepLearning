class Module(object):
    """
        Module class - all other models architecture classes in the framework should inherit from
    """
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []