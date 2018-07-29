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
            Linear(name="linear_1", input_size=784, output_size=10)
        ]
    )




inputs = Variable(name="x")
targets = Variable(name="y")


inputs, targets, loss, grad_node_list = compile(net=model, inputs=inputs, targets=targets, loss=SoftmaxCrossEntropy())

print(loss)

from mnist import *

def one_hot(input):
    values = input
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]

def load_data():
    mnist = {}
    mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"] = load()
    mnist["training_images"] = mnist["training_images"].reshape((60000,28*28))
    mnist["test_images"] = mnist["test_images"].reshape((10000,28*28))
    mnist["training_labels"] = one_hot(mnist["training_labels"])
    mnist["test_labels"] = one_hot(mnist["test_labels"])
    return mnist

dataset = load_data()

train(model, inputs, targets, loss, grad_node_list,
          dataset["training_images"], dataset["training_labels"],
          num_epochs=1000
          )








