import pprint
import random
import torch


class NeuralNetwork:
    def __init__(self, input_size, layers, optimizer, loss, metric=None):
        self.input_size = input_size
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self._layers = []
        self._prediction = []
        self._actual = []
        self._layers.append({
            'a': torch.zeros((self.input_size, 1))
        })
        previous_layer_size = self.input_size
        for layer in self.layers:
            self._layers.append({
                'w': self.__init_weights((layer.size, previous_layer_size)),
                'b': self.__init_biases((layer.size, 1)),
                'z': torch.zeros((layer.size, 1)),
                'activation_function': layer.activation,
                'a': torch.zeros((layer.size, 1))
            })
            previous_layer_size = layer.size

        #print('self._layers')
        #pprint.pprint(self._layers)

    def __init_weights(self, size):
        return torch.tensor([[random.uniform(-0.1, 0.1) for _ in range(size[1])] for _ in range(size[0])])

    def __init_biases(self, size):
        return torch.tensor([[random.uniform(-0.1, 0.1) for _ in range(size[1])] for _ in range(size[0])])

    def __forward(self, layer_input):

        for layer in self._layers[1::]:

            layer['z'] = torch.matmul(layer['w'], layer_input) + layer['b']

            if layer['activation_function'] == 'linear':
                layer['a'] = layer['z']
            else:
                layer['a'] = layer['activation_function'](layer['z'])

            layer_input = layer['a']

        return layer_input

    def __backward(self):
        return None

    def r2_score_manual(self, prediction, actual):

        prediction = torch.flatten(torch.tensor(prediction))
        actual = torch.flatten(torch.tensor(actual))

        ss_res = torch.sum((actual - prediction) ** 2)
        ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)

        r2 = 1.0 - (ss_res / ss_tot)
        return r2

    def fit(self, train_dataset, test_dataset, epochs=10):

        for epoch in range(epochs):

            #print('----------------Epoch ', epoch)

            #debug.append('----------------Epoch ' + str(epoch))

            self._prediction = []
            self._actual = []

            for train_sample in train_dataset:

                input_data = train_sample['input']
                if len(input_data) != self.input_size:
                    raise Exception(f'len(input_data) != self.input_size, len(input_data) = {len(input_data)}, input_data = {input_data}')

                self._layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)

                predict = self.__forward(self._layers[0]['a'])

                self._prediction.append([predict])

                self._actual.append(train_sample['output'])

                self.loss.backward(self.optimizer, self._layers, torch.tensor(predict), torch.tensor(train_sample['output']))

            loss = self.loss.calculate(torch.tensor(self._prediction), torch.tensor(self._actual))
            r2 = self.r2_score_manual(self._prediction, self._actual)

            print(f"Epoch: {epoch}, Loss: {round(float(loss), 4)}, r2 score: {round(float(r2), 4)}")

        self._prediction = []
        self._actual = []

        for test_sample in test_dataset:
            input_data = test_sample['input']
            if len(input_data) != self.input_size:
                raise Exception(
                    f'len(input_data) != self.input_size, len(input_data) = {len(input_data)}, input_data = {input_data}')
            self._layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
            predict = self.__forward(self._layers[0]['a'])
            self._prediction.append([predict])
            self._actual.append(test_sample['output'])

        loss = self.loss.calculate(torch.tensor(self._prediction), torch.tensor(self._actual))
        r2 = self.r2_score_manual(self._prediction, self._actual)

        print(f"Test dataset validation. Loss: {round(float(loss), 4)}, r2 score: {round(float(r2), 4)}")


    def predict(self, data):
        self._prediction = []
        for sample in data:
            input_data = sample['input']
            if len(input_data) != self.input_size:
                raise Exception(
                    f'len(input_data) != self.input_size, len(input_data) = {len(input_data)}, input_data = {input_data}')

            self._layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
            predict = self.__forward(self._layers[0]['a'])
            self._prediction.append([predict])
        return self._prediction
