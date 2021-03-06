import numpy as np
import qnet

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = qnet.autodiff.ops.add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = qnet.autodiff.ops.add_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        """subtracting two nodes return a new node."""
        if isinstance(other, Node):
            new_node = qnet.autodiff.ops.sub_op(self, other)
        else:
            new_node = qnet.autodiff.ops.sub_byconst_op(self, other)
        return new_node


    def __rsub__(self, other):
        """subtracting two nodes return a new node."""
        if isinstance(other, Node):
            new_node = qnet.autodiff.ops.sub_op(other, self)
        else:
            new_node = qnet.autodiff.ops.sub_const_node_op(self, other)
        return new_node

    def __mul__(self, other):
        """TODO: Your code here"""
        if isinstance(other, Node):
            new_node = qnet.autodiff.ops.mul_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = qnet.autodiff.ops.mul_byconst_op(self, other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = qnet.autodiff.ops.div_op(self, other)
        else:
            new_node = qnet.autodiff.ops.div_byconst_op(self, other)
        return new_node

    def __rtruediv__(self, other):
        if isinstance(other, Node):
            new_node = qnet.autodiff.ops.div_op(other, self)
        else:
            new_node = qnet.autodiff.ops.div_const_node_op(other, self)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __str__(self):
        """Allow print to display node name.""" 
        return self.name

    __repr__ = __str__


def Variable(name):
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = qnet.autodiff.ops.placeholder_op()
    placeholder_node.name = name
    return placeholder_node