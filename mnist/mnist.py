import gzip
import struct
import random
import numpy as np



import sys
sys.path.append("../")
from qnet.layer import Linear
from qnet.nn import Sequential
from qnet.train import compile, train
from qnet.autodiff import Variable
from qnet.loss import SoftmaxCrossEntropy
model = Sequential(
        name="net",
        layers = [
            Linear(name="linear_1", input_size=2, output_size=2)
        ]
    )




inputs = Variable(name="x")
targets = Variable(name="y")


inputs, targets, loss, grad_node_list = compile(net=model, inputs=inputs, targets=targets, loss=SoftmaxCrossEntropy())

print(loss)

timage = np.array([
                   [0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1],
                   
                   ], dtype='float32')

tlabel = np.array([
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    
                    ], dtype='float32')
train(model, inputs, targets, loss, grad_node_list,
          timage,
          tlabel,
          num_epochs=800
          )








