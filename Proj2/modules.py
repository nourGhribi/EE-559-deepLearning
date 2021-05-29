import torch
from module import Module
from initializers import Xavier


class Linear(Module):
    """ Fully connected layer
        Parameters : dim_in and dim_out
    """
    def __init__(self, dim_in, dim_out, bias=True, w=None, b=None):
        super(Linear, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.bias = bias
        
        #input
        self.x = None
        
        # Initialize weights with Xavier initalization or with loaded weights
        if w is None:
            self.w = Xavier(self.dim_in,self.dim_out).initialize()
        else:
            self.w = w
            
        if bias:
            if b is None:
                self.b = Xavier(self.dim_out, 1).initialize()
            else:
                self.b = b
            
        # Initialize gradient
        self.dl_dw = torch.empty(self.w.size())
        if bias:
            self.dl_db = torch.empty(self.b.size())
        

    def forward(self, x):
        """ 
        Computes a forward pass 
        :param x: (bs, dim_in), with bs the number of data points (the batch size)
        :return: y = x * W.t + b with W:(dim_out, dim_in), b:(bs, dim_out) returns y:(bs, dim_out)
        """
        self.x = x
        return self.x.mm(self.w.t()) + self.b if self.bias else self.x.mm(self.w.t())  #(batch_size, dim_out)

        
    def backward(self,dl_dz):  
        """ 
        Computes a backward pass and updates gradient of the layer's parameters
        :param dl_dz: gradient with respect to the activation (batch size, dim_out)
        :return: dl_dx: gradient with repect to the input (batchsize, dim_in)
        """
        # z = x * w.t + b
        # l = f(z)
        dz_dx = self.w            # (dim_out, dim_in)
        dl_dx = dl_dz.mm(dz_dx)   # (batchsize, dim_out) * (dim_out, dim_in) = (batchsize,dim_in)
        
        # Update gradients
        if self.bias:
            self.dl_db = dl_dz.sum(0)      #(dim_out, 1)
        self.dl_dw = dl_dz.t().mm(self.x)  #(batchsize, dim_out).T *  (batchsize,dim_in)  = (dim_out , dim_in)
  
        return dl_dx   #(batchsize, dim_in)

    def param(self):
        """
        :return: tuple of tensors of the weights and their respectif gradients
        """
        return [(self.w, self.b, self.dl_dw, self.dl_db)] if self.bias else [(self.w, self.dl_dw)]
    
    def zero_grad(self):
        """
        Set gradient to 0
        """
        self.dl_dw.zero_()
        if self.bias:
            self.dl_db.zero_()


class Sequential(Module):
    """
    Sequential
    """
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        # Take the reverse of the list of layers,
        # and do a backward pass for each layer
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def param(self):
        """
        List of parameters of each layer
        [(self.w, self.b, self.dl_dw, self.dl_db)] or [(self.w, self.dl_dw)] if no bias
        """
        parameters = []
        for layer in self.layers:
            parameters += layer.param()
        return parameters
    
    def __str__(self):            
        params = self.param()
        out = []
        for layer, weights in enumerate(params):
            out.append("layer = "+str(layer+1))
            if len(weights) > 2:
                out.append("w:"+str(params[layer][0].size()))
                out.append("b:"+str(params[layer][1].size()))
            else:
                out.append("w:"+str(params[layer][0].size()))
            out.append("-"*50)
        return "\n".join(out)
    
    def zero_grad(self):
        """
        Set gradients to 0
        """
        for layer in self.layers:
            params = layer.param()
            if len(params) > 0:
                layer.zero_grad()
