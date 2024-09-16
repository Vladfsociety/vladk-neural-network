import random
import time

import torch


class NeuralNetwork:
    """
    Neural Network class for training and prediction.
    """

    def __init__(
        self,
        input_layer,
        layers,
        optimizer,
        loss,
        metric,
        convert_prediction=None,
        use_gpu=False,
        weights_init="uniform",
    ):
        """
        Initialize the neural network with layers, optimizer, loss function, metric, and optional GPU usage.
        """
        if weights_init not in ("uniform", "normal"):
            raise Exception('weights_init value should be "uniform" or "normal"')
        self._use_gpu = use_gpu
        self._weights_init = weights_init
        self._init_device()  # Initialize the device (CPU or GPU)
        self._input_layer = input_layer
        self._optimizer = optimizer
        self._loss = loss
        self._metric = metric
        self._convert_prediction = (
            convert_prediction  # Option for converting prediction to binary or argmax
        )
        self._prediction = []
        self._actual = []
        self._layers = []
        self._init_layers(layers)
        self._optimizer.initialize(self._layers)

    def _init_device(self):
        """
        Set the device to use: GPU if available, otherwise CPU.
        """
        self.device = torch.device(
            "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
        )

    def _init_layers(self, layers):
        """
        Initialize the layers of the neural network.
        Ensure the last layer is fully connected.
        """
        if layers[-1].type != "fully_connected":
            raise Exception("Last layer should be fully connected")

        self._layers.append(self._input_layer.initialize(self.device))
        previous_layer = self._input_layer
        for layer in layers:
            if layer.learnable:
                self._layers.append(
                    layer.initialize(previous_layer, self._weights_init, self.device)
                )
            else:
                self._layers.append(layer.initialize(previous_layer, self.device))
            previous_layer = layer

    def _binary_convert(self, prediction, threshold=0.5):
        """
        Convert prediction to binary values based on a threshold.
        Useful for binary classification tasks.
        """
        return (prediction >= threshold).double()

    def _argmax_convert(self, prediction):
        """
        Convert prediction to one-hot encoded values using argmax.
        Useful for multi-class classification.
        """
        max_indices = torch.argmax(prediction, dim=1, keepdim=True)
        onehot_prediction = torch.zeros_like(prediction)
        onehot_prediction.scatter_(1, max_indices, 1)
        return onehot_prediction

    def _apply_convert_prediction(self, prediction):
        """
        Apply conversion based on the specified prediction type (binary or argmax).
        """
        if self._convert_prediction == "binary":
            prediction = self._binary_convert(prediction)
        elif self._convert_prediction == "argmax":
            prediction = self._argmax_convert(prediction)

        return prediction

    def _forward(self):
        """
        Perform forward pass through the network.
        """
        layer_index = 1

        while layer_index < len(self._layers):
            input_data = self._layers[layer_index - 1].a
            self._layers[layer_index].forward(input_data)
            layer_index += 1

        return self._layers[-1].a

    def _backward(self, predict, actual):
        """
        Perform backward pass to calculate gradients.
        """
        layer_index = len(self._layers) - 1
        layer_error = torch.zeros_like(self._layers[-1].a)

        while layer_index > 0:
            if layer_index == len(self._layers) - 1:
                # Compute loss derivative at the output layer
                loss_derivative = self._loss.derivative(predict, actual)
                layer_error = self._layers[layer_index].backward(
                    loss_derivative,
                    self._layers[layer_index - 1],
                )
            else:
                # Propagate error to the previous layer
                layer_error = self._layers[layer_index].backward(
                    layer_error,
                    self._layers[layer_index - 1],
                    self._layers[layer_index + 1],
                )
            layer_index -= 1

        return

    def _get_sample_input(self, sample):
        input_data = sample["input"]
        if torch.is_tensor(input_data):
            if self._input_layer.type == "input_3d":
                input_data = input_data.to(device=self.device)
            else:
                input_data = input_data.to(device=self.device).reshape(
                    len(input_data), 1
                )
        else:
            if self._input_layer.type == "input_3d":
                input_data = torch.tensor(input_data, device=self.device)
            else:
                input_data = torch.tensor(input_data, device=self.device).reshape(
                    len(input_data), 1
                )

        return input_data

    def _get_sample_output(self, sample):
        output_data = sample["output"]
        if torch.is_tensor(output_data):
            output_data = output_data.to(device=self.device).unsqueeze(1)
        else:
            output_data = torch.tensor(output_data, device=self.device).unsqueeze(1)

        return output_data

    def _process_batch(self, batch):
        """
        Process a batch of data: forward pass, compute loss, and backward pass.
        """
        # Zero out gradients for all learnable layers
        for layer in self._layers[1:]:
            if layer.learnable:
                layer.zero_grad()

        # Loop through each sample in the batch
        for sample in batch:
            input_data = self._get_sample_input(sample)
            output_data = self._get_sample_output(sample)

            self._layers[0].a = input_data

            predict = self._forward()
            self._prediction.append(predict)

            self._actual.append(output_data)

            self._backward(predict, output_data)

        self._optimizer.update(self._layers, len(batch))  # Update the weights

    def fit(
        self, train_dataset, test_dataset=None, epochs=10, batch_size=1, verbose=True
    ):
        """
        Train the neural network for a given number of epochs.
        Optionally evaluate on the test dataset and display progress.
        """
        train_dataset = train_dataset.copy()
        history = []  # Store history of loss and metrics

        for epoch in range(1, epochs + 1):
            start_epoch_time = time.time()
            self._prediction = []
            self._actual = []

            random.shuffle(train_dataset)
            batches = [
                train_dataset[k : k + batch_size]
                for k in range(0, len(train_dataset), batch_size)
            ]

            for batch in batches:
                self._process_batch(batch)

            self._prediction = torch.stack(self._prediction)
            self._actual = torch.stack(self._actual)

            train_loss = self.loss(self._prediction, self._actual)
            train_metric = self.metric(
                self._apply_convert_prediction(self._prediction), self._actual
            )

            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metric": train_metric,
            }

            if test_dataset:
                self._prediction = []
                self._actual = []

                # Evaluate on test dataset
                for test_sample in test_dataset:
                    input_data = self._get_sample_input(test_sample)
                    output_data = self._get_sample_output(test_sample)

                    self._layers[0].a = input_data

                    predict = self._forward()
                    self._prediction.append(predict)

                    self._actual.append(output_data)

                self._prediction = torch.stack(self._prediction)
                self._actual = torch.stack(self._actual)

                # Calculate test loss and metric
                test_loss = self.loss(self._prediction, self._actual)
                test_metric = self.metric(
                    self._apply_convert_prediction(self._prediction), self._actual
                )

                epoch_data["test_loss"] = test_loss
                epoch_data["test_metric"] = test_metric

            epoch_data["epoch_time"] = round(time.time() - start_epoch_time, 3)

            if verbose:
                metric_name = self._metric.name()
                if test_dataset:
                    print(
                        f"Epoch: {epoch_data['epoch']}/{epochs}, "
                        f"train loss: {epoch_data['train_loss']}, "
                        f"train {metric_name}: {epoch_data['train_metric']}, "
                        f"test loss: {epoch_data['test_loss']}, "
                        f"test {metric_name}: {epoch_data['test_metric']}, "
                        f"epoch time: {epoch_data['epoch_time']}s"
                    )
                else:
                    print(
                        f"Epoch: {epoch_data['epoch']}/{epochs}, "
                        f"train loss: {epoch_data['train_loss']}, "
                        f"train {metric_name}: {epoch_data['train_metric']}, "
                        f"epoch time: {epoch_data['epoch_time']}s"
                    )

            history.append(epoch_data)

        return history

    def predict(self, data, with_raw_prediction=False):
        """
        Predict output for the given data.
        Optionally return raw predictions along with the converted ones.
        """
        self._init_device()
        self._prediction = []

        for sample in data:
            input_data = self._get_sample_input(sample)
            self._layers[0].a = input_data

            predict = self._forward()
            self._prediction.append(predict)

        self._prediction = torch.stack(self._prediction)

        if with_raw_prediction:
            return self._apply_convert_prediction(self._prediction), self._prediction
        else:
            return self._apply_convert_prediction(self._prediction)

    def loss(self, prediction, actual):
        """
        Calculate the loss between prediction and actual values.
        """
        return round(float(self._loss.value(prediction, actual.to(self.device))), 4)

    def metric(self, prediction, actual):
        """
        Calculate the metric between prediction and actual values.
        """
        return round(float(self._metric.value(prediction, actual.to(self.device))), 4)
