from qnet.autodiff.ops.op import *
import numpy as np

class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class LogOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents a np.log array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Log(%s)".format(node_A.name)
        return new_node
    def compute(self, node, input_vals):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [(1.0 / node.inputs[0]) * output_grad]

class expOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents a np.exp array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "exp(%s)".format(node_A.name)
        return new_node
    def compute(self, node, input_vals):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.exp(input_vals[0])
    def gradient(self, node, output_grad):
        return [exp_op(node.inputs[0]) * output_grad]

class ReduceSumOp(Op):
    def __call__(self, node_A, axis = 0):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.axis = axis
        new_node.name = "reduce_sum(%s)".format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0], axis=node.axis)

    def gradient(self, node, output_grad):
        return [output_grad]

class MeanOp(Op):
    def __call__(self, node_A, axis = 0):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.axis = axis
        new_node.name = "mean(%s)".format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.mean(input_vals[0], axis=node.axis)

    def gradient(self, node, output_grad):
        return [output_grad]

log_op = LogOp()
exp_op = expOp()
reduce_sum_op = ReduceSumOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
mean_op = MeanOp()