from module import Module

class SGD(Module):

    def __init__(self, model, lr):
        
        super(SGD, self).__init__()
        
        self.model = model
        self.lr = lr
    
    def step(self):
    
        for l in self.model.layers :  
            for tup, _ in enumerate(l.param()): #[(weights, bias, dw, db)] or [(weights, dw)]  for linear module or [] for other modules (ReLU,Sigmoid...)
                if len(tup)>2:
                    l.w = l.w - self.lr * l.dl_dw
                    l.b = l.b - self.lr * l.dl_db
                elif len(tup)==2:
                    l.w = l.w - self.lr * l.dl_dw
                else:
                    raise Exception('Parameters unknown')