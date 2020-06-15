import numpy
from flask import Flask
from flask import request, render_template
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

app = Flask(__name__, static_url_path='', static_folder='templates', template_folder='templates')


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/post', methods=['POST'])
def post_page():
    user_inputs = {key: eval(value) for (key, value) in request.values.dicts[1].items()}

    inputs = numpy.array(user_inputs['input'])
    outputs = numpy.array(user_inputs['output'])
    learning_rate = user_inputs['learning_rate']
    epochs = user_inputs['epochs']

    neural_network = NeuralNetwork(
        [[numpy.array(user_inputs['weights1']), user_inputs['bias1'], activation_functions],
         [numpy.array(user_inputs['weights2']), user_inputs['bias2'], activation_functions]])
    model = Model(neural_network, cost_functions, learning_rate, epochs)
    model.fit(inputs, outputs)
    return "Success"


app.run(port=4995)
