from autodiff.ops.op import *
from autodiff.ops.others import *
import numpy as np


def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax

class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        softmax = softmax_func(y)
        cross_entropy = np.mean(
            -np.sum(y_ * np.log(np.clip(softmax,1e-12,1e100)), axis=1), keepdims=True)
        return cross_entropy

    def gradient(self, node, output_grad):
        grad_A = (softmax_op(node.inputs[0]) + -1 * node.inputs[1])*output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return softmax_func(input_vals[0])

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError




softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmax_op = SoftmaxOp()