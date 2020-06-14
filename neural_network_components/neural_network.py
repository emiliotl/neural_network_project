import numpy
from neural_network_components.layer import Layer


class NeuralNetwork:

    def __init__(self, weights_bias_and_act_functions, number_of_inputs=2):
        self.inputs = number_of_inputs
        self.weights_bias_and_activation_functions = self.__create_array_of_trios(weights_bias_and_act_functions)
        self.neural_network = self.__create_neural_network()

    def __create_array_of_trios(self, weights_bias_and_activation_functions):
        return numpy.append(numpy.array([(self.inputs, None, None)]), weights_bias_and_activation_functions, axis=0)

    def __create_neural_network(self):

        neural_network = []
        for index in range(0, len(self.weights_bias_and_activation_functions) - 1):
            neural_network.append(
                Layer(
                    weights_per_inputs=self.weights_bias_and_activation_functions[index + 1][0],
                    bias_per_neurons=self.weights_bias_and_activation_functions[index + 1][1],
                    activation_functions=self.weights_bias_and_activation_functions[index + 1][2]
                )
            )
        return neural_network
