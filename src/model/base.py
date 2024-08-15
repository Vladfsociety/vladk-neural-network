import random
import torch


class NeuralNetwork:
    def __init__(self, input_layer, layers, optimizer, loss, metric=None):
        self.__input_layer = input_layer
        self.__layers_raw = layers
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
        for layer in self.__layers_raw:
            self.__layers.append(layer.initialize(previous_layer_size))
            previous_layer_size = layer.size

        return None

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

        layer_error = (
            self.__loss.derivative(predict, actual)
            * self.__layers[-1]['activation_function'].derivative(self.__layers[-1]['z'])
        )

        while layer_index > 1:

            grads_w_update[layer_index - 1] = (torch.matmul(layer_error, self.__layers[layer_index - 1]['a'].t()))
            grads_b_update[layer_index - 1] = (torch.tensor(layer_error))

            layer_error = (
                torch.matmul(self.__layers[layer_index]['w'].t(), layer_error)
                * self.__layers[layer_index - 1]['activation_function'].derivative(self.__layers[layer_index - 1]['z'])
            )

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

            self.__actual.append(sample['output'])

            grads_w_update, grads_b_update = self.__backward(torch.tensor(predict), torch.tensor(sample['output']))

            grads_w = [value + update for value, update in zip(grads_w, grads_w_update)]
            grads_b = [value + update for value, update in zip(grads_b, grads_b_update)]

        self.__optimizer.update(self.__layers, grads_w, grads_b, len(batch))

        return None

    def r2_score_manual(self, prediction, actual):

        prediction = torch.flatten(torch.tensor(prediction))
        actual = torch.flatten(torch.tensor(actual))

        ss_res = torch.sum((actual - prediction) ** 2)
        ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)

        r2 = 1.0 - (ss_res / ss_tot)
        return r2

    def fit(self, train_dataset, test_dataset=None, epochs=10, batch_size=1, verbose=False):

        train_dataset = train_dataset.copy()

        for epoch in range(epochs):

            self.__prediction = []
            self.__actual = []

            random.shuffle(train_dataset)
            batches = [train_dataset[k:k + batch_size] for k in range(0, len(train_dataset), batch_size)]

            for batch in batches:
                self.__process_batch(batch)

            if verbose:
                loss = self.__loss.value(torch.tensor(self.__prediction), torch.tensor(self.__actual))
                r2 = self.r2_score_manual(self.__prediction, self.__actual)
                print(f"Epoch: {epoch+1}/{epochs}, Loss: {round(float(loss), 4)}, r2 score: {round(float(r2), 4)}")


        if test_dataset:

            self.__prediction = []
            self.__actual = []

            for test_sample in test_dataset:
                input_data = test_sample['input']
                self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
                predict = self.__forward()
                self.__prediction.append([predict])
                self.__actual.append(test_sample['output'])

            if verbose:
                loss = self.__loss.value(torch.tensor(self.__prediction), torch.tensor(self.__actual))
                r2 = self.r2_score_manual(self.__prediction, self.__actual)
                print(f"Test dataset validation. Loss: {round(float(loss), 4)}, r2 score: {round(float(r2), 4)}")

        return None

    def predict(self, data):
        self.__prediction = []
        for sample in data:
            input_data = sample['input']
            self.__layers[0]['a'] = torch.tensor(input_data).reshape(len(input_data), 1)
            predict = self.__forward()
            self.__prediction.append([predict])
        return self.__prediction
