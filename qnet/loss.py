import qnet.autodiff as ad

class Loss:
    def loss(self,predicted, actual):
        raise NotImplementedError
    

class SoftmaxCrossEntropy(Loss):
    def loss(self,predicted, actual):
        return ad.softmaxcrossentropy_op(predicted, actual)
 