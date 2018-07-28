from qnet.layer import Layer

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


class Sequential(Layer):

    """
        Gather modules as a sequence.
    """
    
    def __init__(self, name, layers):
        super().__init__(name)
        self.layers = layers
    
    def forward(self, inputs, **kwargs):
        training = kwargs['training']
        for layer in self.layers:
            inputs = layer.forward(inputs,training=training)
        return inputs

    def get_params(self):
        params={}
        for layer in self.layers:
            params = merge_two_dicts(params, layer.get_params())

        return params

    def get_all_placeholders(self):
        placeholders = []
        for layer in self.layers:
            placeholders += layer.get_all_placeholders()

        return placeholders

    def get_params_as_list(self):
        params = []
        for layer in self.layers:
            params += layer.get_params_as_list()

        return params

    