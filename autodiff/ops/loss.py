from autodiff.ops.op import *
import numpy as np

class MSE(Op):
	"""Op to calculate Mean Squared Error."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "loss(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return np.sum((input_vals[1]-input_vals[0])**2)

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        # predicted - actual
        grad = input_vals[1]-input_vals[0]
        return [grad, grad]

mse_loss = MSE()