import pprint
import random
import torch


class NeuralNetwork:
    def __init__(self, input_layer, layers, optimizer, loss, metric):
        self.__input_layer = input_layer
        self.__layers_as_objects = layers
        self.__optimizer = optimizer
        self.__loss = loss
        self.__metric = metric
        self.__layers = []
        self.__prediction = []
        self.__actual = []
        self.__init_layers()

    def __init_layers(self):
        self.__layers.append(self.__input_layer.initialize())
        previous_layer_size = self.__input_layer.size
        for layer in self.__layers_as_objects:
            self.__layers.append(layer.initialize(previous_layer_size))
            previous_layer_size = layer.size

        return

    def __forward(self):

        layer_index = 1

        while layer_index < len(self.__layers):

            self.__layers[layer_index]['z'] = (
                torch.matmul(self.__layers[layer_index]['w'], self.__layers[layer_index - 1]['a'])
                + self.__layers[layer_index]['b']
            )
            self.__layers[layer_index]['a'] = (
                self.__layers[layer_index]['activation_function'].apply(self.__layers[layer_index]['z'])
            )

            layer_index += 1

        return self.__layers[-1]['a']

    def __backward(self, predict, actual):

        layer_index = len(self.__layers) - 1

        grads_w_update = [torch.zeros(layer['w'].shape) for layer in self.__layers[1:]]
        grads_b_update = [torch.zeros(layer['b'].shape) for layer in self.__layers[1:]]

        # layer_error = (
        #     self.__loss.derivative(predict, actual)
        #     * self.__layers[-1]['activation_function'].derivative(self.__layers[-1]['z'])
        # )

        layer_error = torch.zeros_like(self.__layers[-1]['a'])

        while layer_index > 0:

            if layer_index == len(self.__layers) - 1:

                # print('self.__loss.derivative(predict, actual)________________________________________________________________________')
                # print(self.__loss.derivative(predict, actual))
                # print('self.__layers[-1][activation_function].derivative(self.__layers[-1][z])')
                # print(self.__layers[-1]['activation_function'].derivative(self.__layers[-1]['z']))

                layer_error = (
                    self.__loss.derivative(predict, actual)
                    * self.__layers[-1]['activation_function'].derivative(self.__layers[-1]['z'])
                )
            else:
                layer_error = (
                    torch.matmul(self.__layers[layer_index + 1]['w'].t(), layer_error)
                    * self.__layers[layer_index]['activation_function'].derivative(self.__layers[layer_index]['z'])
                )

            # print('------------------------------layer_error-----------------------------', layer_index)
            # print(layer_error)

            grads_w_update[layer_index - 1] = (torch.matmul(layer_error, self.__layers[layer_index - 1]['a'].t()))
            grads_b_update[layer_index - 1] = (torch.tensor(layer_error))

            layer_index -= 1

        # print('----------1--------------------layer_error------------------1-----------', layer_index)
        # print(layer_error)
        # grads_w_update[layer_index - 1] = (torch.matmul(layer_error, self.__layers[layer_index - 1]['a'].t()))
        # grads_b_update[layer_index - 1] = (torch.tensor(layer_error))

        # print('___________________grads_w_update')
        # print(grads_w_update)
        # print('___________________grads_b_update')
        # print(grads_b_update)

        return grads_w_update, grads_b_update

    def __process_batch(self, batch):

        grads_w = [torch.zeros(layer['w'].shape) for layer in self.__layers[1:]]
        grads_b = [torch.zeros(layer['b'].shape) for layer in self.__layers[1:]]

        for sample in batch:

            input_data = sample['input']

            self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)

            predict = self.__forward()

            self.__prediction.append([predict])

            self.__actual.append([sample['output']])

            grads_w_update, grads_b_update = self.__backward(torch.tensor(predict), torch.tensor(sample['output']))

            grads_w = [value + update for value, update in zip(grads_w, grads_w_update)]
            grads_b = [value + update for value, update in zip(grads_b, grads_b_update)]

        self.__optimizer.update(self.__layers, grads_w, grads_b, len(batch))

        return

    def fit(self, train_dataset, test_dataset=None, epochs=10, batch_size=1, verbose=False):

        train_dataset = train_dataset.copy()

        # print('self.__layers_before')
        # pprint.pprint(self.__layers)

        for epoch in range(epochs):

            self.__prediction = []
            self.__actual = []

            random.shuffle(train_dataset)
            batches = [train_dataset[k:k + batch_size] for k in range(0, len(train_dataset), batch_size)]

            for batch in batches:
                self.__process_batch(batch)

            if verbose:
                loss = round(float(self.__loss.value(torch.tensor(self.__prediction), torch.tensor(self.__actual))), 4)

                # print('self.__prediction')
                # print(self.__prediction)

                # print('self.__prediction')
                # print(self.__prediction)

                prediction = torch.round(torch.tensor(self.__prediction))

                # print('prediction')
                # print(prediction)

                # print('self.__prediction')
                # print(self.__prediction)

                metric_name = self.__metric.name()
                metric_value = round(float(self.__metric.value(prediction, self.__actual)), 4)
                print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss}, {metric_name}: {metric_value}")

        if test_dataset:

            self.__prediction = []
            self.__actual = []

            for test_sample in test_dataset:
                input_data = test_sample['input']
                self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
                predict = self.__forward()
                self.__prediction.append([predict])
                self.__actual.append([test_sample['output']])

            if verbose:
                loss = round(float(self.__loss.value(torch.tensor(self.__prediction), torch.tensor(self.__actual))), 4)

                # print('self.__prediction')
                # print(self.__prediction)

                prediction = torch.round(torch.tensor(self.__prediction))

                # print('self.__prediction')
                # print(self.__prediction)

                metric_name = self.__metric.name()
                metric_value = round(float(self.__metric.value(prediction, self.__actual)), 4)
                print(f"Test data. Loss: {loss}, {metric_name}: {metric_value}")

        # print('self.__layers_after')
        # pprint.pprint(self.__layers)

        return

    def predict(self, data):
        self.__prediction = []
        for sample in data:
            input_data = sample['input']
            self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
            predict = self.__forward()

            # print('predict_before')
            # print(predict)

            predict = torch.round(predict)

            # print('predict_after')
            # print(predict)

            self.__prediction.append([predict])
        return self.__prediction

    def loss(self, prediction, actual):
        return self.__loss.value(torch.tensor(prediction), torch.tensor(actual))

    def metric(self, prediction, actual):
        return self.__metric.value(prediction, actual)