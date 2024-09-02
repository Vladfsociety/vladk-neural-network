import time
import pprint
import random

import torch

from src.model.layer import Convolutional3x3x16x0x1, Flatten, Input3D


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
        on_cuda=False,
    ):
        self.__input_layer = input_layer
        self.__layers_objects = layers
        self.__optimizer = optimizer
        self.__loss = loss
        self.__metric = metric
        self.__convert_prediction = convert_prediction
        self.__device = torch.device("cuda" if on_cuda and torch.cuda.is_available() else "cpu")
        self.__layers = []
        self.__prediction = []
        self.__actual = []
        self.__init_layers()
        self.__optimizer.initialize(self.__layers)
        print('self.__device')
        print(self.__device)
        # print('self.__layers__start')
        # pprint.pprint(self.__layers)

    def __init_layers(self):
        """
        Initialize the layers of the neural network.
        """
        self.__layers.append(self.__input_layer.initialize(device=self.__device))
        previous_layer = self.__input_layer
        for layer in self.__layers_objects:
            self.__layers.append(layer.initialize(previous_layer, device=self.__device))
            previous_layer = layer

    def __binary_convert(self, prediction, threshold=0.5):
        """
        Convert prediction to binary values based on threshold.
        """
        return (prediction >= threshold).double()

    def __argmax_convert(self, prediction):
        """
        Convert prediction to one-hot encoded values using argmax.
        """
        max_indices = torch.argmax(prediction, dim=1, keepdim=True)
        onehot_prediction = torch.zeros_like(prediction)
        onehot_prediction.scatter_(1, max_indices, 1)
        return onehot_prediction

    def __apply_convert_prediction(self, prediction):
        """
        Apply conversion based on the specified conversion type.
        """
        if self.__convert_prediction == "binary":
            prediction = self.__binary_convert(prediction)
        elif self.__convert_prediction == "argmax":
            prediction = self.__argmax_convert(prediction)

        return prediction

    def __forward(self):
        """
        Perform forward pass through the network.
        """
        layer_index = 1

        start_forward_time = time.time()

        while layer_index < len(self.__layers):

            if self.__layers[layer_index]['type'] == 'convolutional':
                prev_a = self.__layers[layer_index - 1]["a"]

                # print('type(self.__layers[layer_index]["z"])')
                # print(type(self.__layers[layer_index]["z"]))
                # print(self.__layers[layer_index]["z"])

                z = self.__layers[layer_index]["z"].clone()

                # print('prev_a.shape')
                # print(prev_a.shape)
                # print('z.shape_dffdf')
                # print(z.shape)

                k = self.__layers[layer_index]["kernel"]
                input_c = self.__layers[layer_index]["input_c"]
                output_h = self.__layers[layer_index]["output_h"]
                output_w = self.__layers[layer_index]["output_w"]
                filters_num = self.__layers[layer_index]["filters_num"]

                for f in range(filters_num):
                    for i in range(output_h):
                        for j in range(output_w):

                            # print('f ', f)
                            # print('i ', i)
                            # print('j ', j)
                            # print('prev_a[:,i:i+k,j:j+k]')
                            # print(prev_a[:,i:i+k,j:j+k])
                            # print('self.__layers[layer_index][w][f]')
                            # print(self.__layers[layer_index]['w'][f])
                            # print('(torch.sum(prev_a[:,i:i+k,j:j+k] * self.__layers[layer_index][w][f]) + self.__layers[layer_index][b][f])')
                            # print((torch.sum(prev_a[:,i:i+k,j:j+k] * self.__layers[layer_index]['w'][f])
                            #                    + self.__layers[layer_index]['b'][f]))

                            z[f][i][j] = (torch.sum(prev_a[:,i:i+k,j:j+k] * self.__layers[layer_index]['w'][f])
                                               + self.__layers[layer_index]['b'][f])

                self.__layers[layer_index]["z"] = z

                # print('z___________________________________________________________')
                # print(z)

                self.__layers[layer_index]["a"] = self.__layers[layer_index][
                    "activation_function"
                ].apply(self.__layers[layer_index]["z"])

            elif self.__layers[layer_index]['type'] == "flatten":

                # print('self.__layers[layer_index - 1]["a"]')
                # print(self.__layers[layer_index - 1]["a"].shape)
                # print(self.__layers[layer_index - 1]["a"])

                self.__layers[layer_index]["z"] = self.__layers[layer_index - 1]["a"].flatten()

                # print('self.__layers[layer_index]["z"]_before')
                # print(self.__layers[layer_index]["z"].shape)
                # print(self.__layers[layer_index]["z"])

                self.__layers[layer_index]["z"] = self.__layers[layer_index]["z"].reshape(self.__layers[layer_index]["z"].size(0), 1)
                self.__layers[layer_index]["a"] = self.__layers[layer_index]["z"].clone()

                # print('self.__layers[layer_index]["a"]_after')
                # print(self.__layers[layer_index]["a"].shape)
                # print(self.__layers[layer_index]["a"])
            else:

                # print('devices_fdsfsdfds')
                # print(self.__layers[layer_index]['type'])
                # print(self.__layers[layer_index]["w"].device)
                # print(self.__layers[layer_index]["b"].device)
                # print(self.__layers[layer_index]["z"].device)
                # print(self.__layers[layer_index]["a"].device)
                # print(self.__layers[layer_index - 1]["type"])
                # print(self.__layers[layer_index - 1]["a"].device)

                self.__layers[layer_index]["z"] = (
                    torch.matmul(
                        self.__layers[layer_index]["w"], self.__layers[layer_index - 1]["a"]
                    )
                    + self.__layers[layer_index]["b"]
                )

                # print('self.__layers[layer_index]["z"]_fdsfds')
                # print(self.__layers[layer_index]["z"])
                # print('self.__layers[layer_index]["w"]')
                # print(self.__layers[layer_index]["w"])
                # print('self.__layers[layer_index - 1]["a"]')
                # print(self.__layers[layer_index - 1]["a"])
                # print('self.__layers[layer_index]["b"]')
                # print(self.__layers[layer_index]["b"])

                self.__layers[layer_index]["a"] = self.__layers[layer_index][
                    "activation_function"
                ].apply(self.__layers[layer_index]["z"])

            # print('self.__layers[layer_index]["a"]_______')
            # print(self.__layers[layer_index]["a"])

            layer_index += 1

        #print("Forward exec time: ", time.time() - start_forward_time)

        return self.__layers[-1]["a"]

    def __backward(self, predict, actual):
        """
        Perform backward pass to calculate gradients.
        """

        start_backward_time = time.time()

        grads_w_update = [torch.zeros_like(layer["w"]) for layer in self.__layers[1:]]
        grads_b_update = [torch.zeros_like(layer["b"]) for layer in self.__layers[1:]]

        layer_index = len(self.__layers) - 1
        layer_error = torch.zeros_like(self.__layers[-1]["a"])

        while layer_index > 0:

            if layer_index == len(self.__layers) - 1:
                layer_error = self.__loss.derivative(predict, actual) * self.__layers[
                    -1
                ]["activation_function"].derivative(self.__layers[-1]["z"])
            else:

                if self.__layers[layer_index]['type'] == 'flatten': # flatten layer
                    layer_error = torch.matmul(
                        self.__layers[layer_index + 1]["w"].t(), layer_error
                    ) * torch.ones_like(
                        self.__layers[layer_index]["z"]
                    )

                    # print('layer_error_fdsffdsf')
                    # print(layer_error)

                elif self.__layers[layer_index]['type'] == 'convolutional':

                    start_backward_conv_layer = time.time()

                    if self.__layers[layer_index + 1]['type'] == 'flatten':

                        # print('layer_error.shape_before_reshape')
                        # print(layer_error.shape)
                        # print(layer_error)

                        layer_error = layer_error.reshape((
                            self.__layers[layer_index]["output_c"],
                            self.__layers[layer_index]["output_h"],
                            self.__layers[layer_index]["output_w"]
                        ))

                        # print('layer_error.shape_after_reshape')
                        # print(layer_error)

                    elif self.__layers[layer_index + 1]['type'] == 'convolutional':
                        #pass
                        input_h = self.__layers[layer_index]["input_h"]
                        input_w = self.__layers[layer_index]["input_w"]

                        output_h = self.__layers[layer_index]["output_h"]
                        output_w = self.__layers[layer_index]["output_w"]
                        filters_num = self.__layers[layer_index]["filters_num"]

                        k_next = self.__layers[layer_index + 1]["kernel"]
                        output_h_next = self.__layers[layer_index + 1]["output_h"]
                        output_w_next = self.__layers[layer_index + 1]["output_w"]
                        filters_num_next = self.__layers[layer_index + 1]["filters_num"]

                        error = torch.zeros((
                            self.__layers[layer_index]["output_c"],
                            self.__layers[layer_index]["output_h"],
                            self.__layers[layer_index]["output_w"]
                        ), device=self.__device)

                        # print('layer_index')
                        # print(layer_index)
                        # print('error_Fdf')
                        # print(error.shape)

                        # print('layer_error')
                        # print(layer_error.shape)
                        # print('self.__layers[layer_index + 1]["w"]')
                        # print(self.__layers[layer_index + 1]["w"].shape)
                        # print('self.__layers[layer_index]["z"]')
                        # print(self.__layers[layer_index]["z"].shape)

                        # print('layer_error.device')
                        # print(layer_error.device)

                        #layer_error_padding = layer_error.clone()

                        #layer_error_padding = []

                        # print('layer_error_before_fdsfd')
                        # print(layer_error)

                        pad_value = 0.0

                        P_top, P_bottom, P_left, P_right = [k_next - 1]*4
                        #H, W = layer_error.shape

                        # Create a new array with the padded shape
                        padded_H = output_h_next + P_top + P_bottom
                        padded_W = output_w_next + P_left + P_right


                        # Copy the original tensor into the center of the padded tensor
                        # for filter_next in range(filters_num_next):
                        #     padded = torch.full((padded_H, padded_W), pad_value)
                        #     padded[P_top:P_top + output_h_next, P_left:P_left + output_w_next] = layer_error[filter_next].clone()
                        #     layer_error_padding.append(padded)
                        #
                        # print('layer_error_padding_Fdf')
                        # print(layer_error_padding)
                        #
                        # layer_error_padding = torch.tensor(layer_error_padding)
                        # print('layer_error_padding')
                        # print(layer_error_padding)
                        layer_error_with_padding = torch.full((filters_num_next, padded_H, padded_W), pad_value, device=self.__device)
                        layer_error_with_padding[:,P_top:P_top + output_h_next, P_left:P_left + output_w_next] = layer_error[:].clone()

                        # print('layer_error_with_padding')
                        # print(layer_error_with_padding)

                        # print('self.__layers[layer_index + 1]["w"]')
                        # print(self.__layers[layer_index + 1]["w"])

                        flipped_w_next = torch.flip(self.__layers[layer_index + 1]["w"], (2, 3))

                        # print('flipped_w_next.device')
                        # print(flipped_w_next.device)

                        # print('flipped_w_next')
                        # print(flipped_w_next)
                        #
                        # import sys
                        # sys.exit(1)

                        for f in range(filters_num):
                            for i in range(output_h):
                                for j in range(output_w):

                                    # for f_next in range(filters_num_next):
                                    #
                                    #     # print('self.__layers[layer_index + 1]["w"][f_next][f]')
                                    #     # print(self.__layers[layer_index + 1]["w"][f_next][f])
                                    #     flipped_w_next = torch.flip(self.__layers[layer_index + 1]["w"][f_next][f], (0, 1))
                                    #     # print('flipped_w_next')
                                    #     # print(flipped_w_next)
                                    #     #
                                    #     # print('layer_error_with_padding[f_next][i:i + k_next, j:j + k_next]')
                                    #     # print(layer_error_with_padding[f_next][i:i + k_next, j:j + k_next])
                                    #
                                    #     error[f][i][j] += (torch.sum(
                                    #         layer_error_with_padding[f_next][i:i + k_next, j:j + k_next] * flipped_w_next))

                                    error[f][i][j] = (torch.sum(
                                                layer_error_with_padding[:, i:i + k_next, j:j + k_next] * flipped_w_next[:, f]))

                                    error[f][i][j] = (error[f][i][j]
                                                     * self.__layers[layer_index]['activation_function'].derivative(self.__layers[layer_index]["z"][f][i][j]))
                                    # print('-----------------f, i, j - ', f, i, j)
                                    # print('error[f][i][j]_before')
                                    # print(error[f][i][j])
                                    #
                                    # for f_next in range(filters_num_next):
                                    #
                                    #     i_bot = (i-(k_next-1)) if (i-(k_next-1) > 0) else 0
                                    #     i_top = i+1 if i+1 < output_h_next else output_h_next
                                    #     j_bot = (j-(k_next-1)) if (j-(k_next-1) > 0) else 0
                                    #     j_top = j+1 if j+1 < output_w_next else output_w_next
                                    #
                                    #     #m_bot =
                                    #     print('i_bot')
                                    #     print(i_bot)
                                    #     print('i_top')
                                    #     print(i_top)
                                    #     print('j_bot')
                                    #     print(j_bot)
                                    #     print('j_top')
                                    #     print(j_top)
                                    #
                                    #     print('layer_error[f_next][i_bot:i_top,j_bot:j_top]')
                                    #     print(layer_error[f_next][
                                    #           i_bot:i_top,
                                    #           j_bot:j_top
                                    #           ])
                                    #     print('self.__layers[layer_index + 1]["w"][f_next][f][0:(k_next-1),0:(k_next-1)]')
                                    #     print(self.__layers[layer_index + 1]["w"][f_next][f][
                                    #             0:k_next,
                                    #             0:k_next
                                    #         ])
                                    #
                                    #     error[f][i][j] = (torch.sum(layer_error[f_next][
                                    #             i_bot:i_top,
                                    #             j_bot:j_top
                                    #         ] * self.__layers[layer_index + 1]["w"][f_next][f][
                                    #             0:k_next,
                                    #             0:k_next
                                    #         ])
                                    #         * self.__layers[layer_index]['activation_function'].derivative(
                                    #             self.__layers[layer_index]["z"][f][i][j]
                                    #         )
                                    #     )
                                    #
                                    #     # error[f][i][j] = (torch.sum(layer_error[f_next].t()[
                                    #     #                             i_bot:i_top,
                                    #     #                             j_bot:j_top
                                    #     #                             ] * self.__layers[layer_index + 1]["w"][f_next][f][
                                    #     #                                 0:k_next,
                                    #     #                                 0:k_next
                                    #     #                                 ])
                                    #     #                   * self.__layers[layer_index][
                                    #     #                       'activation_function'].derivative(
                                    #     #             self.__layers[layer_index]["z"][f][i][j]
                                    #     #         )
                                    #     #                   )
                                    #
                                    # error[f][i][j] = (error[f][i][j]
                                    #                 * self.__layers[layer_index]['activation_function'].derivative(self.__layers[layer_index]["z"][f][i][j]))

                                    # for ii in range(k_next):
                                    #     for jj in range(k_next):
                                    #         for f_next in range(filters_num_next):
                                    #
                                    #             # print('f, i, ii, j, jj - ', f, i, ii, j, jj)
                                    #             # print('i - ii - ', i - ii)
                                    #             # print('j - jj - ', j - jj)
                                    #
                                    #             if (
                                    #                 (0 <= i - ii < output_h_next)
                                    #                 and (0 <= j - jj < output_w_next)
                                    #             ):
                                    #
                                    #                 # error[f][i][j] += (layer_error[f_next][i - ii][j - jj]
                                    #                 #        * self.__layers[layer_index + 1]["w"][f_next][f][ii][jj]
                                    #                 #        * self.__layers[layer_index]['activation_function'].derivative(self.__layers[layer_index]["z"][f][i][j])) # Умножение на єту хрень думаю можно делать один раз в другом месте
                                    #                 error[f][i][j] += (layer_error[f_next][i - ii][j - jj]
                                    #                                    * self.__layers[layer_index + 1]["w"][f_next][f][ii][jj])
                                    #
                                    # error[f][i][j] = (error[f][i][j]
                                    #                   * self.__layers[layer_index]['activation_function'].derivative(self.__layers[layer_index]["z"][f][i][j]))

                                    # print('error[f][i][j]_after')
                                    # print(error[f][i][j])
                                    # print('i-(k-1) : i ', i-(k-1), i)
                                    # print('j-(k-1) : j ', j-(k-1), j)
                                    # print('layer_error')
                                    # print(layer_error[f_next][
                                    #         i-(k-1):i,
                                    #         j-(k-1):j
                                    #     ].shape)
                                    # print(layer_error[f_next][
                                    #       i - (k - 1):i,
                                    #       j - (k - 1):j
                                    #       ])
                                    # print('self.__layers[layer_index + 1]["w"]')
                                    # print(self.__layers[layer_index + 1]["w"][f_next][f][
                                    #         0:(k-1),
                                    #         0:(k-1)
                                    #     ].shape)
                                    # print(self.__layers[layer_index + 1]["w"][f_next][f][
                                    #       0:(k - 1),
                                    #       0:(k - 1)
                                    #       ])
                                    # print('self.__layers[layer_index]["z"]')
                                    # print(self.__layers[layer_index]["z"][f][i][j].shape)
                                    # print(self.__layers[layer_index]["z"][f][i][j])
                                    #
                                    # error[f][i][j] = (torch.sum(layer_error[f_next][
                                    #         i-(k-1):i,
                                    #         j-(k-1):j
                                    #     ] * self.__layers[layer_index + 1]["w"][f_next][f][
                                    #         0:(k-1),
                                    #         0:(k-1)
                                    #     ])
                                    #     * self.__layers[layer_index]['activation_function'].derivative(
                                    #         self.__layers[layer_index]["z"][f][i][j]
                                    #     )
                                    # )
                                    #
                                    # print('error[f][i][j]')
                                    # print(error[f][i][j])

                        # print('error_AFTER')
                        # print(type(error))
                        # print(error)

                        # layer_error = torch.tensor(error)
                        layer_error = error

                    #print('Backward conv layer exec: ', time.time() - start_backward_conv_layer)

                else:
                    layer_error = torch.matmul(
                        self.__layers[layer_index + 1]["w"].t(), layer_error
                    ) * self.__layers[layer_index]["activation_function"].derivative(
                        self.__layers[layer_index]["z"]
                    )

            # print('__________________________________________________self.__layers[layer_index - 1]')
            # print(self.__layers[layer_index - 1])
            # print('layer_error')
            # print(layer_error.shape)
            # print(layer_error)
            # print('self.__layers[layer_index - 1]["a"].t()')
            # print(self.__layers[layer_index - 1]["a"])

            if self.__layers[layer_index]['type'] == 'flatten':
                grads_w_update[layer_index - 1] = torch.tensor(0)
                grads_b_update[layer_index - 1] = torch.tensor(0)

            elif self.__layers[layer_index]['type'] == 'convolutional':

                # print('layer_error_convolutional')
                # print(layer_error.shape)
                # print(layer_error)

                k = self.__layers[layer_index]["kernel"]
                filters_num = self.__layers[layer_index]["filters_num"]
                input_c = self.__layers[layer_index]["input_c"]
                output_h = self.__layers[layer_index]["output_h"]
                output_w = self.__layers[layer_index]["output_w"]

                # print('input_c')
                # print(input_c)
                # print('output_h')
                # print(output_h)
                # print('output_w')
                # print(output_w)
                # print('output_w')
                # print(output_w)
                # print('layer_error.shape')
                # print(layer_error.shape)

                for f in range(filters_num):
                    for c in range(input_c):
                        for m in range(k):
                            for n in range(k):

                                # print('m ', m)
                                # print('n ', n)
                                # print('self.__layers[layer_index - 1]["a"][m:m+output_h:,n:n+output_w:]')
                                # print(self.__layers[layer_index - 1]["a"][
                                #         c,
                                #         m:m+output_h:,
                                #         n:n+output_w:
                                #     ].shape)
                                # print(self.__layers[layer_index - 1]["a"][
                                #         c,
                                #         m:m+output_h:,
                                #         n:n+output_w:
                                #     ])
                                # print('layer_error[f]')
                                # print(layer_error[f])
                                # print(layer_error[f] * self.__layers[layer_index - 1]["a"][
                                #         c,
                                #         m:m+output_h:,
                                #         n:n+output_w:
                                #     ])
                                # print(layer_error[f].shape)
                                # print(layer_error[f])

                                grads_w_update[layer_index - 1][f][c][m][n] = torch.sum(
                                    layer_error[f] * self.__layers[layer_index - 1]["a"][
                                        c,
                                        m:m+output_h:,
                                        n:n+output_w:
                                    ]
                                )

                                # print('grads_w_update[layer_index - 1]')
                                # print(grads_w_update[layer_index - 1].shape)
                                # print(grads_w_update[layer_index - 1])

                    # print('torch.sum(layer_error[f])')
                    # print(torch.sum(layer_error[f]))

                    grads_b_update[layer_index - 1][f] = torch.sum(layer_error[f]) # ???

                # print('grads_w_update[layer_index - 1]')
                # print(grads_w_update[layer_index - 1].shape)
                # print(grads_w_update[layer_index - 1])
                # print('grads_b_update[layer_index - 1]')
                # print(grads_b_update[layer_index - 1].shape)
                # print(grads_b_update[layer_index - 1])

            else:
                grads_w_update[layer_index - 1] = torch.matmul(
                    layer_error, self.__layers[layer_index - 1]["a"].t()
                )
                grads_b_update[layer_index - 1] = layer_error

            layer_index -= 1

        #print("Backward exec time: ", time.time() - start_backward_time)

        return grads_w_update, grads_b_update

    def __process_batch(self, batch):
        """
        Process a batch of data.
        """
        grads_w = [torch.zeros_like(layer["w"]) for layer in self.__layers[1:]]
        grads_b = [torch.zeros_like(layer["b"]) for layer in self.__layers[1:]]

        for sample in batch:
            input_data = sample["input"]

            # print('self.__layers_objects[0]')
            # print(self.__layers_objects[0])

            if isinstance(self.__input_layer, Input3D):
                self.__layers[0]["a"] = torch.tensor(input_data, device=self.__device)
            else:
                self.__layers[0]["a"] = torch.tensor(input_data, device=self.__device).reshape(len(input_data), 1)

            predict = self.__forward()

            self.__prediction.append(predict)

            output = torch.tensor(sample["output"], device=self.__device).unsqueeze(1)

            self.__actual.append(output)

            grads_w_update, grads_b_update = self.__backward(predict, output)

            grads_w = [value + update for value, update in zip(grads_w, grads_w_update)]
            grads_b = [value + update for value, update in zip(grads_b, grads_b_update)]

        self.__optimizer.update(self.__layers, grads_w, grads_b, len(batch))

    def fit(
        self, train_dataset, test_dataset=None, epochs=10, batch_size=1, verbose=True
    ):
        """
        Train the neural network on the provided dataset.
        """
        train_dataset = train_dataset.copy()

        print('len(test_dataset)')
        print(len(test_dataset))

        history = []

        for epoch in range(1, epochs + 1):

            start_epoch_time = time.time()

            self.__prediction = []
            self.__actual = []

            # if epoch % 50 == 0:
            #     print('self.__layers')
            #     print(self.__layers)

            random.shuffle(train_dataset)
            batches = [
                train_dataset[k : k + batch_size]
                for k in range(0, len(train_dataset), batch_size)
            ]

            for batch in batches:
                self.__process_batch(batch)

            self.__prediction = torch.stack(self.__prediction)
            self.__actual = torch.stack(self.__actual)

            train_loss = self.loss(self.__prediction, self.__actual)
            train_metric = self.metric(
                self.__apply_convert_prediction(self.__prediction), self.__actual
            )

            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metric": train_metric,
            }

            if test_dataset:
                self.__prediction = []
                self.__actual = []

                for test_sample in test_dataset:
                    input_data = test_sample["input"]

                    if isinstance(self.__input_layer, Input3D):
                        self.__layers[0]["a"] = torch.tensor(input_data, device=self.__device)
                    else:
                        self.__layers[0]["a"] = torch.tensor(input_data, device=self.__device).reshape(
                            len(input_data), 1
                        )

                    # self.__layers[0]["a"] = torch.tensor(input_data).reshape(
                    #     len(input_data), 1
                    # )

                    predict = self.__forward()

                    self.__prediction.append(predict)
                    self.__actual.append(
                        torch.tensor(test_sample["output"], device=self.__device).unsqueeze(1)
                    )

                self.__prediction = torch.stack(self.__prediction)
                self.__actual = torch.stack(self.__actual)

                # print('self.__prediction')
                # print(self.__prediction)
                # print('self.__apply_convert_prediction(self.__prediction)')
                # print(self.__apply_convert_prediction(self.__prediction))
                # print('self.__actual')
                # print(self.__actual)

                test_loss = self.loss(self.__prediction, self.__actual)
                test_metric = self.metric(
                    self.__apply_convert_prediction(self.__prediction), self.__actual
                )

                epoch_data["test_loss"] = test_loss
                epoch_data["test_metric"] = test_metric

            if verbose:
                metric_name = self.__metric.name()
                if test_dataset:
                    print(
                        f"Epoch: {epoch_data['epoch']}/{epochs}, "
                        f"train loss: {epoch_data['train_loss']}, "
                        f"train {metric_name}: {epoch_data['train_metric']}, "
                        f"test loss: {epoch_data['test_loss']}, "
                        f"test {metric_name}: {epoch_data['test_metric']}"
                    )
                else:
                    print(
                        f"Epoch: {epoch_data['epoch']}/{epochs}, "
                        f"train loss: {epoch_data['train_loss']}, "
                        f"train {metric_name}: {epoch_data['train_metric']}"
                    )

            print('Epoch time: ', time.time() - start_epoch_time)

            history.append(epoch_data)

        # print('self.__layers__after')
        # pprint.pprint(self.__layers)

        return history

    def predict(self, data, with_raw_prediction=False):
        """
        Predict output for the given data.
        """
        self.__prediction = []

        for sample in data:
            input_data = sample["input"]

            if isinstance(self.__input_layer, Input3D):
                self.__layers[0]["a"] = torch.tensor(input_data, device=self.__device)
            else:
                self.__layers[0]["a"] = torch.tensor(input_data, device=self.__device).reshape(
                    len(input_data), 1
                )

            #self.__layers[0]["a"] = torch.tensor(input_data).reshape(len(input_data), 1)

            predict = self.__forward()

            self.__prediction.append(predict)

        self.__prediction = torch.stack(self.__prediction)

        if with_raw_prediction:
            return self.__apply_convert_prediction(self.__prediction), self.__prediction
        else:
            return self.__apply_convert_prediction(self.__prediction)

    def loss(self, prediction, actual):
        """
        Calculate the loss between prediction and actual values.
        """
        return round(float(self.__loss.value(prediction, actual)), 4)

    def metric(self, prediction, actual):
        """
        Calculate the metric between prediction and actual values.
        """
        return round(float(self.__metric.value(prediction, actual)), 4)
