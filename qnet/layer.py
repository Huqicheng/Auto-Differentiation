from typing import Dict,Tuple
from qnet.tensor import Tensor
import qnet.autodiff as ad
import numpy as np

class Layer:
    def __init__(self, name):
        self.placeholders = {}
        self.params = {}
        self.name = name

    def forward(self, inputs, **kwargs):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def get_all_placeholders(self):
        placeholders = []
        for name, placeholder in self.placeholders.items():
            placeholders.append(placeholder)
        return placeholders

    def get_params_as_list(self):
        params = []
        for name, param in self.params.items():
            params.append(param)

        return params



def linear(x, W, b): 
    return ad.mat_add_vector_op(ad.matmul_op(x, W),b)

class Linear(Layer):

    def __init__(self, 
                 name,
                 input_size,
                 output_size):
        super().__init__(name)
        self.placeholders["w"] = ad.Variable(name=self.name+"_w")
        self.placeholders["b"] = ad.Variable(name=self.name+"_b")
        self.params["w"] = np.float32(np.random.randn(input_size,output_size))
        self.params["b"] = np.float32(np.random.randn(output_size))

    def forward(self, inputs, **kwargs):
        return linear(inputs, self.placeholders["w"], self.placeholders["b"])

    def get_params(self):
        return {
            self.placeholders["w"] : self.params["w"],
            self.placeholders["b"] : self.params["b"]
        }
