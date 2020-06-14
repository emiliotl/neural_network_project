import numpy


class Model:

    def __init__(self, neural_network, cost_functions, learning_rate, epochs, steps):
        self.neural_network = neural_network.neural_network
        self.cost_functions = cost_functions
        self.learning_rate = learning_rate
        self.loss = []
        self.epochs = epochs
        self.steps = steps

    def fit(self, inputs, outputs):
        for i in range(self.epochs):
            self.__forward_pass(inputs)
            self.__backward_pass(outputs)
        print(self.loss[-1])
        print(self.predict(inputs))

    def predict(self, inputs):
        input_series = [inputs]
        for layer in self.neural_network:
            net = input_series[-1] @ layer.weights.T + layer.bias
            output = layer.activation_functions[0](net)
            input_series.append(output)
        return input_series[-1]

    def __forward_pass(self, inputs):
        self.input_series = [inputs]
        for layer in self.neural_network:
            net = self.input_series[-1] @ layer.weights.T + layer.bias
            output = layer.activation_functions[0](net)
            self.input_series.append(output)

    def __backward_pass(self, outputs):
        self.deltas = []
        loss = self.cost_functions[0](self.input_series[-1], outputs)
        self.loss.append(loss)

        for index in reversed(range(0, len(self.neural_network))):
            inputs = self.input_series[index + 1]

            if index == len(self.neural_network) - 1:
                self.deltas.insert(0, self.cost_functions[1](outputs, inputs) *
                                   self.neural_network[index].activation_functions[1](inputs))
                previous_weights_previous_layer = numpy.copy(self.neural_network[index].weights.T)
                self.neural_network[index].weights = (self.neural_network[index].weights.T - self.input_series[index] * self.deltas[0] * self.learning_rate).T
            else:
                self.deltas.insert(0, self.deltas[0] @ previous_weights_previous_layer[index] *
                                   self.neural_network[index].activation_functions[1](inputs))
                previous_weights_previous_layer = numpy.copy(self.neural_network[index].weights.T)
                self.neural_network[index].weights = (self.neural_network[index].weights.T - self.input_series[0] * self.deltas[0] * self.learning_rate).T
