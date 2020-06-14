class Layer:

    def __init__(self, weights_per_inputs, bias_per_neurons, activation_functions):
        self.weights = weights_per_inputs
        self.bias = bias_per_neurons
        self.activation_functions = activation_functions
