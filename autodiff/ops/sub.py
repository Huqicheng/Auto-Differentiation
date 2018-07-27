from autodiff.ops.op import *
import numpy as np

class SubOp(Op):
    """Op to element-wise sub two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise subtraction."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of sub node, return gradient contributions to each input."""
        return [output_grad, -1.0 * output_grad]


class SubByConstOp(Op):
    """Op to element-wise sub a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise subtraction."""
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of sub node, return gradient contribution to input."""
        return [output_grad]

class SubConstByNodeOp(Op):
    """Op to element-wise sub a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise subtraction."""
        assert len(input_vals) == 1
        return node.const_attr - input_vals[0]

    def gradient(self, node, output_grad):
        """Given gradient of sub node, return gradient contribution to input."""
        return [-1.0 * output_grad]



sub_byconst_op = SubByConstOp()
sub_const_node_op = SubConstByNodeOp()
sub_op = SubOp()