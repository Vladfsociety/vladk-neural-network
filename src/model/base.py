import time
import pprint
import random
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
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._input_layer = input_layer
        self._optimizer = optimizer
        self._loss = loss
        self._metric = metric
        self._convert_prediction = convert_prediction
        self._prediction = []
        self._actual = []
        self._layers = []
        self._init_layers(layers)
        self._optimizer.initialize(self._layers)

    def _init_layers(self, layers):
        """
        Initialize the layers of the neural network.
        """
        if layers[-1].type != "fully_connected":
            raise Exception("Last layer should be fully connected")

        self._layers.append(self._input_layer.initialize(self.device))
        previous_layer = self._input_layer
        for layer in layers:
            self._layers.append(layer.initialize(previous_layer, self.device))
            previous_layer = layer

    def _binary_convert(self, prediction, threshold=0.5):
        """
        Convert prediction to binary values based on threshold.
        """
        return (prediction >= threshold).double()

    def _argmax_convert(self, prediction):
        """
        Convert prediction to one-hot encoded values using argmax.
        """
        max_indices = torch.argmax(prediction, dim=1, keepdim=True)
        onehot_prediction = torch.zeros_like(prediction)
        onehot_prediction.scatter_(1, max_indices, 1)
        return onehot_prediction

    def _apply_convert_prediction(self, prediction):
        """
        Apply conversion based on the specified conversion type.
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
            input = self._layers[layer_index - 1].a

            # print('self._layers[layer_index].name')
            # print(self._layers[layer_index].name)
            # start_time = time.time()

            self._layers[layer_index].forward(input)

            #print("Forward time: ", time.time() - start_time)

            layer_index += 1

        return self._layers[-1].a

    def _backward(self, predict, actual):
        """
        Perform backward pass to calculate gradients.
        """
        layer_index = len(self._layers) - 1
        layer_error = torch.zeros_like(self._layers[-1].a)

        while layer_index > 0:

            # print('self._layers[layer_index].name')
            # print(self._layers[layer_index].name)
            # start_time = time.time()

            if layer_index == len(self._layers) - 1:
                loss_derivative = self._loss.derivative(predict, actual)
                layer_error = self._layers[layer_index].backward(
                    loss_derivative,
                    self._layers[layer_index - 1],
                )
            else:
                layer_error = self._layers[layer_index].backward(
                    layer_error,
                    self._layers[layer_index - 1],
                    self._layers[layer_index + 1]
                )

            #print("Backward time: ", time.time() - start_time)

            layer_index -= 1

        return

    # def _forward(self):
    #     """
    #     Perform forward pass through the network.
    #     """
    #     layer_index = 1
    #
    #     start_forward_time = time.time()
    #
    #     while layer_index < len(self._layers):
    #
    #         input = self._layers[layer_index - 1].a
    #
    #         self._layers[layer_index].forward(input)
    #
    #         if self._layers[layer_index]['type'] == 'convolutional':
    #             prev_a = self._layers[layer_index - 1]["a"]
    #
    #             # print('type(self._layers[layer_index]["z"])')
    #             # print(type(self._layers[layer_index]["z"]))
    #             # print(self._layers[layer_index]["z"])
    #
    #             z = self._layers[layer_index]["z"].clone()
    #
    #             # print('prev_a.shape')
    #             # print(prev_a.shape)
    #             # print('z.shape_dffdf')
    #             # print(z.shape)
    #
    #             k = self._layers[layer_index]["kernel"]
    #             input_c = self._layers[layer_index]["input_c"]
    #             output_h = self._layers[layer_index]["output_h"]
    #             output_w = self._layers[layer_index]["output_w"]
    #             filters_num = self._layers[layer_index]["filters_num"]
    #
    #             for f in range(filters_num):
    #                 for i in range(output_h):
    #                     for j in range(output_w):
    #
    #                         # print('f ', f)
    #                         # print('i ', i)
    #                         # print('j ', j)
    #                         # print('prev_a[:,i:i+k,j:j+k]')
    #                         # print(prev_a[:,i:i+k,j:j+k])
    #                         # print('self._layers[layer_index][w][f]')
    #                         # print(self._layers[layer_index]['w'][f])
    #                         # print('(torch.sum(prev_a[:,i:i+k,j:j+k] * self._layers[layer_index][w][f]) + self._layers[layer_index][b][f])')
    #                         # print((torch.sum(prev_a[:,i:i+k,j:j+k] * self._layers[layer_index]['w'][f])
    #                         #                    + self._layers[layer_index]['b'][f]))
    #
    #                         z[f][i][j] = (torch.sum(prev_a[:,i:i+k,j:j+k] * self._layers[layer_index]['w'][f])
    #                                            + self._layers[layer_index]['b'][f])
    #
    #             self._layers[layer_index]["z"] = z
    #
    #             # print('z___________________________________________________________')
    #             # print(z)
    #
    #             self._layers[layer_index]["a"] = self._layers[layer_index][
    #                 "activation_function"
    #             ].apply(self._layers[layer_index]["z"])
    #
    #             # print('self._layers[layer_index]["a"][0]')
    #             # self.plot_digit(self._layers[layer_index]["a"][0])
    #             #
    #             # print('self._layers[layer_index]["a"][1]')
    #             # self.plot_digit(self._layers[layer_index]["a"][1])
    #
    #         elif self._layers[layer_index]['type'] == "flatten":
    #
    #             # print('self._layers[layer_index - 1]["a"]')
    #             # print(self._layers[layer_index - 1]["a"].shape)
    #             # print(self._layers[layer_index - 1]["a"])
    #
    #             self._layers[layer_index]["z"] = self._layers[layer_index - 1]["a"].flatten()
    #
    #             # print('self._layers[layer_index]["z"]_before')
    #             # print(self._layers[layer_index]["z"].shape)
    #             # print(self._layers[layer_index]["z"])
    #
    #             self._layers[layer_index]["z"] = self._layers[layer_index]["z"].reshape(self._layers[layer_index]["z"].size(0), 1)
    #             self._layers[layer_index]["a"] = self._layers[layer_index]["z"].clone()
    #
    #             # print('self._layers[layer_index]["a"]_after')
    #             # print(self._layers[layer_index]["a"].shape)
    #             # print(self._layers[layer_index]["a"])
    #         else:
    #
    #             # print('devices_fdsfsdfds')
    #             # print(self._layers[layer_index]['type'])
    #             # print(self._layers[layer_index]["w"].device)
    #             # print(self._layers[layer_index]["b"].device)
    #             # print(self._layers[layer_index]["z"].device)
    #             # print(self._layers[layer_index]["a"].device)
    #             # print(self._layers[layer_index - 1]["type"])
    #             # print(self._layers[layer_index - 1]["a"].device)
    #
    #             self._layers[layer_index]["z"] = (
    #                 torch.matmul(
    #                     self._layers[layer_index]["w"], self._layers[layer_index - 1]["a"]
    #                 )
    #                 + self._layers[layer_index]["b"]
    #             )
    #
    #             # print('self._layers[layer_index]["z"]_fdsfds')
    #             # print(self._layers[layer_index]["z"])
    #             # print('self._layers[layer_index]["w"]')
    #             # print(self._layers[layer_index]["w"])
    #             # print('self._layers[layer_index - 1]["a"]')
    #             # print(self._layers[layer_index - 1]["a"])
    #             # print('self._layers[layer_index]["b"]')
    #             # print(self._layers[layer_index]["b"])
    #
    #             self._layers[layer_index]["a"] = self._layers[layer_index][
    #                 "activation_function"
    #             ].apply(self._layers[layer_index]["z"])
    #
    #         # print('self._layers[layer_index]["a"]_______')
    #         # print(self._layers[layer_index]["a"])
    #
    #         layer_index += 1
    #
    #     #print("Forward exec time: ", time.time() - start_forward_time)
    #
    #     return self._layers[-1].a

    # def _backward(self, predict, actual):
    #     """
    #     Perform backward pass to calculate gradients.
    #     """
    #
    #     start_backward_time = time.time()
    #
    #     grads_w_update = [torch.zeros_like(layer.w) for layer in self._layers[1:]]
    #     grads_b_update = [torch.zeros_like(layer.b) for layer in self._layers[1:]]
    #
    #     layer_index = len(self._layers) - 1
    #     layer_error = torch.zeros_like(self._layers[-1]["a"])
    #
    #     while layer_index > 0:
    #
    #         if layer_index == len(self._layers) - 1:
    #
    #             loss_derivative = self._loss.derivative(predict, actual)
    #
    #             layer_error = self._layers[layer_index].backward(loss_derivative)
    #         else:
    #
    #             if self._layers[layer_index]['type'] == 'flatten': # flatten layer
    #                 layer_error = torch.matmul(
    #                     self._layers[layer_index + 1]["w"].t(), layer_error
    #                 ) * torch.ones_like(
    #                     self._layers[layer_index]["z"]
    #                 )
    #
    #                 # print('layer_error_fdsffdsf')
    #                 # print(layer_error)
    #
    #             elif self._layers[layer_index]['type'] == 'convolutional':
    #
    #                 start_backward_conv_layer = time.time()
    #
    #                 if self._layers[layer_index + 1]['type'] == 'flatten':
    #
    #                     # print('layer_error.shape_before_reshape')
    #                     # print(layer_error.shape)
    #                     # print(layer_error)
    #
    #                     layer_error = layer_error.reshape((
    #                         self._layers[layer_index]["output_c"],
    #                         self._layers[layer_index]["output_h"],
    #                         self._layers[layer_index]["output_w"]
    #                     ))
    #
    #                     # print('layer_error.shape_after_reshape')
    #                     # print(layer_error)
    #
    #                 elif self._layers[layer_index + 1]['type'] == 'convolutional':
    #                     #pass
    #                     input_h = self._layers[layer_index]["input_h"]
    #                     input_w = self._layers[layer_index]["input_w"]
    #
    #                     output_h = self._layers[layer_index]["output_h"]
    #                     output_w = self._layers[layer_index]["output_w"]
    #                     filters_num = self._layers[layer_index]["filters_num"]
    #
    #                     k_next = self._layers[layer_index + 1]["kernel"]
    #                     output_h_next = self._layers[layer_index + 1]["output_h"]
    #                     output_w_next = self._layers[layer_index + 1]["output_w"]
    #                     filters_num_next = self._layers[layer_index + 1]["filters_num"]
    #
    #                     error = torch.zeros((
    #                         self._layers[layer_index]["output_c"],
    #                         self._layers[layer_index]["output_h"],
    #                         self._layers[layer_index]["output_w"]
    #                     ), device=self.device)
    #
    #                     # print('layer_index')
    #                     # print(layer_index)
    #                     # print('error_Fdf')
    #                     # print(error.shape)
    #
    #                     # print('layer_error')
    #                     # print(layer_error.shape)
    #                     # print('self._layers[layer_index + 1]["w"]')
    #                     # print(self._layers[layer_index + 1]["w"].shape)
    #                     # print('self._layers[layer_index]["z"]')
    #                     # print(self._layers[layer_index]["z"].shape)
    #
    #                     # print('layer_error.device')
    #                     # print(layer_error.device)
    #
    #                     #layer_error_padding = layer_error.clone()
    #
    #                     #layer_error_padding = []
    #
    #                     # print('layer_error_before_fdsfd')
    #                     # print(layer_error)
    #
    #                     pad_value = 0.0
    #
    #                     P_top, P_bottom, P_left, P_right = [k_next - 1]*4
    #                     #H, W = layer_error.shape
    #
    #                     # Create a new array with the padded shape
    #                     padded_H = output_h_next + P_top + P_bottom
    #                     padded_W = output_w_next + P_left + P_right
    #
    #
    #                     # Copy the original tensor into the center of the padded tensor
    #                     # for filter_next in range(filters_num_next):
    #                     #     padded = torch.full((padded_H, padded_W), pad_value)
    #                     #     padded[P_top:P_top + output_h_next, P_left:P_left + output_w_next] = layer_error[filter_next].clone()
    #                     #     layer_error_padding.append(padded)
    #                     #
    #                     # print('layer_error_padding_Fdf')
    #                     # print(layer_error_padding)
    #                     #
    #                     # layer_error_padding = torch.tensor(layer_error_padding)
    #                     # print('layer_error_padding')
    #                     # print(layer_error_padding)
    #                     layer_error_with_padding = torch.full((filters_num_next, padded_H, padded_W), pad_value, device=self.device)
    #                     layer_error_with_padding[:,P_top:P_top + output_h_next, P_left:P_left + output_w_next] = layer_error[:].clone()
    #
    #                     # print('layer_error_with_padding')
    #                     # print(layer_error_with_padding)
    #
    #                     # print('self._layers[layer_index + 1]["w"]')
    #                     # print(self._layers[layer_index + 1]["w"])
    #
    #                     flipped_w_next = torch.flip(self._layers[layer_index + 1]["w"], (2, 3))
    #
    #                     # print('flipped_w_next.device')
    #                     # print(flipped_w_next.device)
    #
    #                     # print('flipped_w_next')
    #                     # print(flipped_w_next)
    #                     #
    #                     # import sys
    #                     # sys.exit(1)
    #
    #                     for f in range(filters_num):
    #                         for i in range(output_h):
    #                             for j in range(output_w):
    #
    #                                 # for f_next in range(filters_num_next):
    #                                 #
    #                                 #     # print('self._layers[layer_index + 1]["w"][f_next][f]')
    #                                 #     # print(self._layers[layer_index + 1]["w"][f_next][f])
    #                                 #     flipped_w_next = torch.flip(self._layers[layer_index + 1]["w"][f_next][f], (0, 1))
    #                                 #     # print('flipped_w_next')
    #                                 #     # print(flipped_w_next)
    #                                 #     #
    #                                 #     # print('layer_error_with_padding[f_next][i:i + k_next, j:j + k_next]')
    #                                 #     # print(layer_error_with_padding[f_next][i:i + k_next, j:j + k_next])
    #                                 #
    #                                 #     error[f][i][j] += (torch.sum(
    #                                 #         layer_error_with_padding[f_next][i:i + k_next, j:j + k_next] * flipped_w_next))
    #
    #                                 error[f][i][j] = (torch.sum(
    #                                             layer_error_with_padding[:, i:i + k_next, j:j + k_next] * flipped_w_next[:, f]))
    #
    #                                 error[f][i][j] = (error[f][i][j]
    #                                                  * self._layers[layer_index]['activation_function'].derivative(self._layers[layer_index]["z"][f][i][j]))
    #                                 # print('-----------------f, i, j - ', f, i, j)
    #                                 # print('error[f][i][j]_before')
    #                                 # print(error[f][i][j])
    #                                 #
    #                                 # for f_next in range(filters_num_next):
    #                                 #
    #                                 #     i_bot = (i-(k_next-1)) if (i-(k_next-1) > 0) else 0
    #                                 #     i_top = i+1 if i+1 < output_h_next else output_h_next
    #                                 #     j_bot = (j-(k_next-1)) if (j-(k_next-1) > 0) else 0
    #                                 #     j_top = j+1 if j+1 < output_w_next else output_w_next
    #                                 #
    #                                 #     #m_bot =
    #                                 #     print('i_bot')
    #                                 #     print(i_bot)
    #                                 #     print('i_top')
    #                                 #     print(i_top)
    #                                 #     print('j_bot')
    #                                 #     print(j_bot)
    #                                 #     print('j_top')
    #                                 #     print(j_top)
    #                                 #
    #                                 #     print('layer_error[f_next][i_bot:i_top,j_bot:j_top]')
    #                                 #     print(layer_error[f_next][
    #                                 #           i_bot:i_top,
    #                                 #           j_bot:j_top
    #                                 #           ])
    #                                 #     print('self._layers[layer_index + 1]["w"][f_next][f][0:(k_next-1),0:(k_next-1)]')
    #                                 #     print(self._layers[layer_index + 1]["w"][f_next][f][
    #                                 #             0:k_next,
    #                                 #             0:k_next
    #                                 #         ])
    #                                 #
    #                                 #     error[f][i][j] = (torch.sum(layer_error[f_next][
    #                                 #             i_bot:i_top,
    #                                 #             j_bot:j_top
    #                                 #         ] * self._layers[layer_index + 1]["w"][f_next][f][
    #                                 #             0:k_next,
    #                                 #             0:k_next
    #                                 #         ])
    #                                 #         * self._layers[layer_index]['activation_function'].derivative(
    #                                 #             self._layers[layer_index]["z"][f][i][j]
    #                                 #         )
    #                                 #     )
    #                                 #
    #                                 #     # error[f][i][j] = (torch.sum(layer_error[f_next].t()[
    #                                 #     #                             i_bot:i_top,
    #                                 #     #                             j_bot:j_top
    #                                 #     #                             ] * self._layers[layer_index + 1]["w"][f_next][f][
    #                                 #     #                                 0:k_next,
    #                                 #     #                                 0:k_next
    #                                 #     #                                 ])
    #                                 #     #                   * self._layers[layer_index][
    #                                 #     #                       'activation_function'].derivative(
    #                                 #     #             self._layers[layer_index]["z"][f][i][j]
    #                                 #     #         )
    #                                 #     #                   )
    #                                 #
    #                                 # error[f][i][j] = (error[f][i][j]
    #                                 #                 * self._layers[layer_index]['activation_function'].derivative(self._layers[layer_index]["z"][f][i][j]))
    #
    #                                 # for ii in range(k_next):
    #                                 #     for jj in range(k_next):
    #                                 #         for f_next in range(filters_num_next):
    #                                 #
    #                                 #             # print('f, i, ii, j, jj - ', f, i, ii, j, jj)
    #                                 #             # print('i - ii - ', i - ii)
    #                                 #             # print('j - jj - ', j - jj)
    #                                 #
    #                                 #             if (
    #                                 #                 (0 <= i - ii < output_h_next)
    #                                 #                 and (0 <= j - jj < output_w_next)
    #                                 #             ):
    #                                 #
    #                                 #                 # error[f][i][j] += (layer_error[f_next][i - ii][j - jj]
    #                                 #                 #        * self._layers[layer_index + 1]["w"][f_next][f][ii][jj]
    #                                 #                 #        * self._layers[layer_index]['activation_function'].derivative(self._layers[layer_index]["z"][f][i][j])) # Умножение на єту хрень думаю можно делать один раз в другом месте
    #                                 #                 error[f][i][j] += (layer_error[f_next][i - ii][j - jj]
    #                                 #                                    * self._layers[layer_index + 1]["w"][f_next][f][ii][jj])
    #                                 #
    #                                 # error[f][i][j] = (error[f][i][j]
    #                                 #                   * self._layers[layer_index]['activation_function'].derivative(self._layers[layer_index]["z"][f][i][j]))
    #
    #                                 # print('error[f][i][j]_after')
    #                                 # print(error[f][i][j])
    #                                 # print('i-(k-1) : i ', i-(k-1), i)
    #                                 # print('j-(k-1) : j ', j-(k-1), j)
    #                                 # print('layer_error')
    #                                 # print(layer_error[f_next][
    #                                 #         i-(k-1):i,
    #                                 #         j-(k-1):j
    #                                 #     ].shape)
    #                                 # print(layer_error[f_next][
    #                                 #       i - (k - 1):i,
    #                                 #       j - (k - 1):j
    #                                 #       ])
    #                                 # print('self._layers[layer_index + 1]["w"]')
    #                                 # print(self._layers[layer_index + 1]["w"][f_next][f][
    #                                 #         0:(k-1),
    #                                 #         0:(k-1)
    #                                 #     ].shape)
    #                                 # print(self._layers[layer_index + 1]["w"][f_next][f][
    #                                 #       0:(k - 1),
    #                                 #       0:(k - 1)
    #                                 #       ])
    #                                 # print('self._layers[layer_index]["z"]')
    #                                 # print(self._layers[layer_index]["z"][f][i][j].shape)
    #                                 # print(self._layers[layer_index]["z"][f][i][j])
    #                                 #
    #                                 # error[f][i][j] = (torch.sum(layer_error[f_next][
    #                                 #         i-(k-1):i,
    #                                 #         j-(k-1):j
    #                                 #     ] * self._layers[layer_index + 1]["w"][f_next][f][
    #                                 #         0:(k-1),
    #                                 #         0:(k-1)
    #                                 #     ])
    #                                 #     * self._layers[layer_index]['activation_function'].derivative(
    #                                 #         self._layers[layer_index]["z"][f][i][j]
    #                                 #     )
    #                                 # )
    #                                 #
    #                                 # print('error[f][i][j]')
    #                                 # print(error[f][i][j])
    #
    #                     # print('error_AFTER')
    #                     # print(type(error))
    #                     # print(error)
    #
    #                     # layer_error = torch.tensor(error)
    #                     layer_error = error
    #
    #                 #print('Backward conv layer exec: ', time.time() - start_backward_conv_layer)
    #
    #             else:
    #                 layer_error = torch.matmul(
    #                     self._layers[layer_index + 1]["w"].t(), layer_error
    #                 ) * self._layers[layer_index]["activation_function"].derivative(
    #                     self._layers[layer_index]["z"]
    #                 )
    #
    #         # print('__________________________________________________self._layers[layer_index - 1]')
    #         # print(self._layers[layer_index - 1])
    #         # print('layer_error')
    #         # print(layer_error.shape)
    #         # print(layer_error)
    #         # print('self._layers[layer_index - 1]["a"].t()')
    #         # print(self._layers[layer_index - 1]["a"])
    #
    #         if self._layers[layer_index]['type'] == 'flatten':
    #             grads_w_update[layer_index - 1] = torch.tensor(0)
    #             grads_b_update[layer_index - 1] = torch.tensor(0)
    #
    #         elif self._layers[layer_index]['type'] == 'convolutional':
    #
    #             # print('layer_error_convolutional')
    #             # print(layer_error.shape)
    #             # print(layer_error)
    #
    #             k = self._layers[layer_index]["kernel"]
    #             filters_num = self._layers[layer_index]["filters_num"]
    #             input_c = self._layers[layer_index]["input_c"]
    #             output_h = self._layers[layer_index]["output_h"]
    #             output_w = self._layers[layer_index]["output_w"]
    #
    #             # print('input_c')
    #             # print(input_c)
    #             # print('output_h')
    #             # print(output_h)
    #             # print('output_w')
    #             # print(output_w)
    #             # print('output_w')
    #             # print(output_w)
    #             # print('layer_error.shape')
    #             # print(layer_error.shape)
    #
    #             for f in range(filters_num):
    #                 for c in range(input_c):
    #                     for m in range(k):
    #                         for n in range(k):
    #
    #                             # print('m ', m)
    #                             # print('n ', n)
    #                             # print('self._layers[layer_index - 1]["a"][m:m+output_h:,n:n+output_w:]')
    #                             # print(self._layers[layer_index - 1]["a"][
    #                             #         c,
    #                             #         m:m+output_h:,
    #                             #         n:n+output_w:
    #                             #     ].shape)
    #                             # print(self._layers[layer_index - 1]["a"][
    #                             #         c,
    #                             #         m:m+output_h:,
    #                             #         n:n+output_w:
    #                             #     ])
    #                             # print('layer_error[f]')
    #                             # print(layer_error[f])
    #                             # print(layer_error[f] * self._layers[layer_index - 1]["a"][
    #                             #         c,
    #                             #         m:m+output_h:,
    #                             #         n:n+output_w:
    #                             #     ])
    #                             # print(layer_error[f].shape)
    #                             # print(layer_error[f])
    #
    #                             grads_w_update[layer_index - 1][f][c][m][n] = torch.sum(
    #                                 layer_error[f] * self._layers[layer_index - 1]["a"][
    #                                     c,
    #                                     m:m+output_h:,
    #                                     n:n+output_w:
    #                                 ]
    #                             )
    #
    #                             # print('grads_w_update[layer_index - 1]')
    #                             # print(grads_w_update[layer_index - 1].shape)
    #                             # print(grads_w_update[layer_index - 1])
    #
    #                 # print('torch.sum(layer_error[f])')
    #                 # print(torch.sum(layer_error[f]))
    #
    #                 grads_b_update[layer_index - 1][f] = torch.sum(layer_error[f]) # ???
    #
    #             # print('grads_w_update[layer_index - 1]')
    #             # print(grads_w_update[layer_index - 1].shape)
    #             # print(grads_w_update[layer_index - 1])
    #             # print('grads_b_update[layer_index - 1]')
    #             # print(grads_b_update[layer_index - 1].shape)
    #             # print(grads_b_update[layer_index - 1])
    #
    #         else:
    #             grads_w_update[layer_index - 1] = torch.matmul(
    #                 layer_error, self._layers[layer_index - 1]["a"].t()
    #             )
    #             grads_b_update[layer_index - 1] = layer_error
    #
    #         layer_index -= 1
    #
    #     #print("Backward exec time: ", time.time() - start_backward_time)
    #
    #     return grads_w_update, grads_b_update

    def _process_batch(self, batch):
        """
        Process a batch of data.
        """
        for layer in self._layers[1:]:
            if layer.learnable:
                layer.zero_grad()

        for sample in batch:
            input_data = sample["input"]

            if self._input_layer.type == 'input_3d':
                self._layers[0].a = torch.tensor(input_data, device=self.device)
            else:
                self._layers[0].a = torch.tensor(input_data, device=self.device).reshape(len(input_data), 1)

            predict = self._forward()
            self._prediction.append(predict)

            output = torch.tensor(sample["output"], device=self.device).unsqueeze(1)
            self._actual.append(output)

            self._backward(predict, output)

        self._optimizer.update(self._layers, len(batch))

    def fit(
        self, train_dataset, test_dataset=None, epochs=10, batch_size=1, verbose=True
    ):
        """
        Train the neural network on the provided dataset.
        """
        train_dataset = train_dataset.copy()

        history = []

        for epoch in range(1, epochs + 1):

            start_epoch_time = time.time()

            self._prediction = []
            self._actual = []

            # if epoch % 50 == 0:
            #     print('self._layers')
            #     print(self._layers)

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

                for test_sample in test_dataset:
                    input_data = test_sample["input"]

                    if self._input_layer.type == 'input_3d':
                        self._layers[0].a = torch.tensor(input_data, device=self.device)
                    else:
                        self._layers[0].a = torch.tensor(input_data, device=self.device).reshape(
                            len(input_data), 1
                        )

                    # self._layers[0]["a"] = torch.tensor(input_data).reshape(
                    #     len(input_data), 1
                    # )

                    predict = self._forward()
                    self._prediction.append(predict)
                    self._actual.append(
                        torch.tensor(test_sample["output"], device=self.device).unsqueeze(1)
                    )

                self._prediction = torch.stack(self._prediction)
                self._actual = torch.stack(self._actual)

                # print('self._prediction')
                # print(self._prediction)
                # print('self._apply_convert_prediction(self._prediction)')
                # print(self._apply_convert_prediction(self._prediction))
                # print('self._actual')
                # print(self._actual)

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
                        f"epoch time: {epoch_data["epoch_time"]}s"
                    )
                else:
                    print(
                        f"Epoch: {epoch_data['epoch']}/{epochs}, "
                        f"train loss: {epoch_data['train_loss']}, "
                        f"train {metric_name}: {epoch_data['train_metric']}, "
                        f"epoch time: {epoch_data["epoch_time"]}s"
                    )

            history.append(epoch_data)

        # print('self._layers__after')
        # pprint.pprint(self._layers)

        return history

    def predict(self, data, with_raw_prediction=False):
        """
        Predict output for the given data.
        """
        self._prediction = []

        for sample in data:
            input_data = sample["input"]

            if self._input_layer.type == 'input_3d':
                self._layers[0].a = torch.tensor(input_data, device=self.device)
            else:
                self._layers[0].a = torch.tensor(input_data, device=self.device).reshape(
                    len(input_data), 1
                )

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
        return round(float(self._loss.value(prediction, actual)), 4)

    def metric(self, prediction, actual):
        """
        Calculate the metric between prediction and actual values.
        """
        return round(float(self._metric.value(prediction, actual)), 4)
