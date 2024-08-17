import pprint
import random
import torch


class NeuralNetwork:
    def __init__(self, input_layer, layers, optimizer, loss, metric, convert_prediction=None):
        self.__input_layer        = input_layer
        self.__layers_objects     = layers
        self.__optimizer          = optimizer
        self.__loss               = loss
        self.__metric             = metric
        self.__convert_prediction = convert_prediction
        self.__layers             = []
        self.__prediction         = []
        self.__actual             = []
        self.__init_layers()

    def __init_layers(self):
        self.__layers.append(self.__input_layer.initialize())
        previous_layer_size = self.__input_layer.size
        for layer in self.__layers_objects:
            self.__layers.append(layer.initialize(previous_layer_size))
            previous_layer_size = layer.size

        return

    def __binary_convert(self, prediction):
        threshold = 0.5
        return (prediction >= threshold).float()

    def __apply_convert_prediction(self, prediction):

        if self.__convert_prediction == 'binary':
            prediction = self.__binary_convert(prediction)

        return prediction

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

        return self.__layers[-1]['a'].flatten()

    def __backward(self, predict, actual):

        grads_w_update = [torch.zeros(layer['w'].shape) for layer in self.__layers[1:]]
        grads_b_update = [torch.zeros(layer['b'].shape) for layer in self.__layers[1:]]

        layer_index = len(self.__layers) - 1
        layer_error = torch.zeros_like(self.__layers[-1]['a'])

        while layer_index > 0:

            if layer_index == len(self.__layers) - 1:
                layer_error = (
                    self.__loss.derivative(predict, actual)
                    * self.__layers[-1]['activation_function'].derivative(self.__layers[-1]['z'])
                )
            else:
                layer_error = (
                    torch.matmul(self.__layers[layer_index + 1]['w'].t(), layer_error)
                    * self.__layers[layer_index]['activation_function'].derivative(self.__layers[layer_index]['z'])
                )

            grads_w_update[layer_index - 1] = (torch.matmul(layer_error, self.__layers[layer_index - 1]['a'].t()))
            grads_b_update[layer_index - 1] = layer_error

            layer_index -= 1

        return grads_w_update, grads_b_update

    def __process_batch(self, batch):

        grads_w = [torch.zeros(layer['w'].shape) for layer in self.__layers[1:]]
        grads_b = [torch.zeros(layer['b'].shape) for layer in self.__layers[1:]]

        for sample in batch:

            input_data = sample['input']

            self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)

            predict = self.__forward()

            self.__prediction.append([predict])

            output = torch.tensor(sample['output'])

            self.__actual.append([output])

            grads_w_update, grads_b_update = self.__backward(predict, output)

            grads_w = [value + update for value, update in zip(grads_w, grads_w_update)]
            grads_b = [value + update for value, update in zip(grads_b, grads_b_update)]

        self.__optimizer.update(self.__layers, grads_w, grads_b, len(batch))

        return

    def fit(self, train_dataset, test_dataset=None, epochs=10, batch_size=1, verbose=True):

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

            self.__prediction = torch.tensor(self.__prediction)
            self.__actual = torch.tensor(self.__actual)

            if verbose:
                loss = self.loss(self.__prediction, self.__actual)
                metric_name = self.__metric.name()
                metric_value = self.metric(self.__apply_convert_prediction(self.__prediction), self.__actual)
                print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss}, {metric_name}: {metric_value}")

        if test_dataset and verbose:

            self.__prediction = []
            self.__actual = []

            for test_sample in test_dataset:
                input_data = test_sample['input']
                self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
                predict = self.__forward()
                self.__prediction.append([predict])
                self.__actual.append([torch.tensor(test_sample['output'])])

            self.__prediction = torch.tensor(self.__prediction)
            self.__actual = torch.tensor(self.__actual)

            loss = self.loss(self.__prediction, self.__actual)
            metric_name = self.__metric.name()
            metric_value = self.metric(self.__apply_convert_prediction(self.__prediction), self.__actual)
            print(f"Test dataset. Loss: {loss}, {metric_name}: {metric_value}")

        # print('self.__layers_after')
        # pprint.pprint(self.__layers)

        return

    def predict(self, data):

        self.__prediction = []

        for sample in data:
            input_data = sample['input']
            self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
            predict = self.__forward()

            self.__prediction.append([predict])

        return self.__apply_convert_prediction(torch.tensor(self.__prediction))

    def loss(self, prediction, actual):
        return round(float(self.__loss.value(prediction, actual)), 4)

    def metric(self, prediction, actual):
        return round(float(self.__metric.value(prediction, actual)), 4)