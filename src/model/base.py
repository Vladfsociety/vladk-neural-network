import pprint
import torch


class NeuralNetwork:
    def __init__(self, input_size, layers, optimizer, loss=None, metric=None):
        self.input_size = input_size
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self._layers = []
        self._prediction = []
        self._actual = []
        self._loss = None
        print(self.layers)
        previous_layer_size = self.input_size
        for layer in self.layers:
            self._layers.append({
                'input': torch.zeros(previous_layer_size),
                'weights': torch.zeros(previous_layer_size, layer.size),
                'biases': torch.zeros(layer.size),
                'activation': layer.activation,
                'output': torch.zeros(layer.size)
            })
            previous_layer_size = layer.size

        print('self._layers')
        pprint.pprint(self._layers)

    def __forward(self, layer_input):

        for index, layer in enumerate(self._layers):
            print('------------------Layer index ', index)

            layer['input'] = layer_input

            sums = (torch.torch.matmul(layer['input'], layer['weights'])
                    + layer['biases'])

            if layer['activation'] == 'linear':
                results = sums
            else:
                results = layer['activation'](sums)

            layer['output'] = results

            layer_input = layer['output']

        return layer_input

    def __backward(self):

        print('self._loss')
        print(self._loss)

        return None

    def fit(self, train_dataset, epochs=1):

        self._prediction = []
        self._actual = []

        for epoch in range(epochs):

            print('----------------Epoch ', epoch)

            for train_data in train_dataset:

                layer_input = torch.tensor(train_data['input'])

                self._actual.append(train_data['output'])

                predict = self.__forward(layer_input)

                self._prediction.append([predict])

            self._loss = self.loss.calculate(torch.tensor(self._prediction), torch.tensor(self._actual))

            self.optimizer.update(self._layers, self._loss)

            #self.__backward()

