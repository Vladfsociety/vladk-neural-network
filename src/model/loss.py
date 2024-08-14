import pprint
import torch


class MeanSquaredError:
    def __init__(self):
        self.learning_rate = 0.001
        self._gradients = []
        self._errors = []
        pass

    def calculate(self, predicted, actual):
        return ((actual - predicted)**2).sum()/predicted.size(0)

    def loss_derivative(self, predicted, actual):
        return predicted - actual

    def backward(self, optimizer, layers, predicted, actual):

        layer_error = self.loss_derivative(predicted, actual) * 1

        layer_index = len(layers) - 1
        for layer in layers[-1:1:-1]:

            # print('layer_index')
            # print(layer_index)
            # print('layer_error')
            # pprint.pprint(layer_error)
            # print('layers[layer_index - 1][a]')
            # pprint.pprint(layers[layer_index - 1]['a'])

            layers[layer_index]['w'] -= self.learning_rate * torch.matmul(layer_error, layers[layer_index - 1]['a'].t())
            layers[layer_index]['b'] -= self.learning_rate * layer_error

            def der_relu(z):
                return torch.where(z > 0.0, 1.0, 0.0)

            def der_sigmoid(z):
                return z*(torch.ones((z.size(0), 1)) - z)

            layer_error = torch.matmul(layer['w'].t(), layer_error) * der_relu(layers[layer_index - 1]['z'])
            layer_index -= 1
