from module import Module

class LossMSE(Module):
    
    def __init__(self,model):
        """
        :param: the model for which to compute the loss on its output
        """
        super(LossMSE, self).__init__()
        self.prediction = None
        self.target = None
        self.model = model
        
    def forward(self, prediction, target):
        """
        :param prediction: the predicted values using the model
        :param target: the true values
        :return: MSE error over the batch size
        """
        self.prediction = prediction
        self.target = target
        batched_error = (prediction - target).pow(2).sum(1) #L2 norm of the difference between prediction and target
        return batched_error.mean(0)  #take the mean over the batch
    
    def backward(self):
        batchsize = self.prediction.shape[0]
        dMSE = 2*(self.prediction - self.target)/batchsize
        #proagate the loss to the model
        self.model.backward(dMSE)