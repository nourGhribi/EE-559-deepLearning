class Initializer(object):
    """
        Initializer class - all other initializers architecture classes in the framework should inherit from.
    """
    
    def __init__(self):
        pass
    
    def initialize(self,tensor):
        raise NotImplementedError
