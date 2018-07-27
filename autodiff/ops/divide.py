from autodiff.ops.op import *
import numpy as np

class DivOp(Op):
    """op for perfoming element-wise division"""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)".format(node_A, node_B)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [1.0 / node.inputs[1], -1 * (node.inputs[0] / (node.inputs[1] * node.inputs[1]))]

class DivByConstOp(Op):
    """Op to element-wise divide by a constant"""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s/%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise division."""
        assert len(input_vals) == 1
        return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of division node, return gradient contribution to input."""
        """TODO: Your code here"""
        return [output_grad / node.const_attr]

class DivConstByNodeOP(Op):
    """Op to element-wise divison of constant by a node"""
    def __call__(self, const_val, node_A):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s/%s)" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise division by constant."""
        assert len(input_vals) == 1
        return node.const_attr / input_vals[0]

    def gradient(self, node, output_grad):
        """Given gradient of division node, return gradient contribution to input."""
        return [(-1.0 * node.const_attr / (node.inputs[0] * node.inputs[0])) * output_grad]

div_op = DivOp()
div_byconst_op = DivByConstOp()
div_const_node_op = DivConstByNodeOP()
