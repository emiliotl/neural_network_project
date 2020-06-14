import numpy
from neural_network_components.neural_network import NeuralNetwork
from neural_network_components.model import Model


def cost_function(expected_output, actual_output):
    return 0.5 * sum((expected_output - actual_output) ** 2)


def derivative_cost_function(expected_output, actual_output):
    return actual_output - expected_output


def activation_function(input_value):
    result = []
    for single_input in input_value:
        result.append(0.1*max(0, single_input))
    return numpy.array(result)


def derivative_activation_function(input_value):
    result = []
    for single_input in input_value:
        if single_input >= 0:
            result.append(0.1)
        else:
            result.append(0)
    return numpy.array(result)


cost_functions = (cost_function, derivative_cost_function)
activation_functions = (activation_function, derivative_activation_function)

inputs = numpy.array([2, 3])
outputs = numpy.array([60, 40])
learning_rate = 0.01
epochs = 100000
steps = 500

neural_network = NeuralNetwork(
    [[numpy.array([[1, 2], [3, 2]]), 0.35, activation_functions],
     [numpy.array([[2, 3], [2, 1]]), 0.60, activation_functions]])
model = Model(neural_network, cost_functions, learning_rate, epochs, steps)
model.fit(inputs, outputs)
