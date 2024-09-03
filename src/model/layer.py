import math
import random
import time

import torch
import torch.multiprocessing as mp


class Layer:
    def _init_weights(self, size):
        n_inputs = size[0]
        limit = math.sqrt(1 / n_inputs)
        return torch.tensor(
            [
                [random.uniform(-limit, limit) for _ in range(size[1])]
                for _ in range(size[0])
            ]
        )

    def _init_biases(self, size):
        return torch.zeros(size)


class Input(Layer):
    def __init__(self, size):
        self.type = "input"
        self.size = size
        self.a = torch.zeros((self.size, 1))


class Input3D(Layer):
    def __init__(self, size):
        self.type = "input_3d"
        self.size = size
        self.a = torch.zeros((self.size[0], self.size[1], self.size[2]))


class FullyConnected(Layer):
    def __init__(self, size, activation):
        self.type = "fully_connected"
        self.learnable = True
        self.size = size
        self.activation = activation

    def _calculate_layer_error(self, next_layer_error, next_layer):
        if not next_layer:
            layer_error = next_layer_error * self.activation.derivative(self.z)
        else:
            layer_error = (torch.matmul(next_layer.w.t(), next_layer_error)
                           * self.activation.derivative(self.z))

        return layer_error

    def initialize(self, previous_layer):
        previous_layer_size = previous_layer.size
        self.z = torch.zeros((self.size, 1))
        self.a = torch.zeros((self.size, 1))
        self.w = super()._init_weights((self.size, previous_layer_size))
        self.b = super()._init_biases((self.size, 1))
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)

        return self

    def zero_grad(self):
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)
        return

    def forward(self, input):
        self.z = torch.matmul(self.w, input) + self.b
        self.a = self.activation.apply(self.z)
        return

    def backward(self, next_layer_error, prev_layer, next_layer=None):
        layer_error = self._calculate_layer_error(next_layer_error, next_layer)

        self.grad_w += torch.matmul(layer_error, prev_layer.a.t())
        self.grad_b += layer_error

        return layer_error


class Convolutional(Layer):
    def __init__(self, activation, filters_num=4, kernel_size=3, padding=0, stride=1):
        self.type = "convolutional"
        self.learnable = True
        self.activation = activation
        self.filters_num = filters_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def _init_weights(self, size):
        n_inputs = size[1] * size[2] * size[3]
        random.seed(41)

        limit = math.sqrt(1 / n_inputs)
        return torch.tensor(
            [
                [[[random.uniform(-limit, limit) for _ in range(size[3])]
                for _ in range(size[2])]
                for _ in range(size[1])]
                for _ in range(size[0])
            ]
        )

    def _calculate_layer_error(self, next_layer_error, next_layer):
        if next_layer.type == "convolutional":
            k_next = next_layer.kernel_size
            layer_error_with_padding = self._get_padded_tensor(next_layer_error, [k_next-1]*4)
            layer_error = torch.zeros((self.output_c, self.output_h, self.output_w))
            flipped_w_next = torch.flip(next_layer.w, (2, 3))
            for f in range(self.filters_num):
                for i in range(self.output_h):
                    for j in range(self.output_w):
                        layer_error[f][i][j] = (
                            torch.sum(
                                layer_error_with_padding[:, i:i + k_next, j:j + k_next]
                                * flipped_w_next[:, f]
                            )
                        )
            layer_error *= self.activation.derivative(self.z)
        else:
            layer_error = next_layer_error

        return layer_error

    def _get_padded_tensor(self, tensor, padding, padding_value=0.0):
        p_top, p_bot, p_left, p_right = padding
        f, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2]

        padded_h = h + p_top + p_bot
        padded_w = w + p_left + p_right

        padded_tensor = torch.full((f, padded_h, padded_w), padding_value)
        padded_tensor[:, p_top:p_top + h, p_left:p_left + w] = tensor[:].clone()

        return padded_tensor

    def initialize(self, previous_layer):
        if previous_layer.type == 'convolutional':
            self.input_c = previous_layer.output_c
            self.input_h = previous_layer.output_h
            self.input_w = previous_layer.output_w
        else:
            self.input_c = previous_layer.size[0]
            self.input_h = previous_layer.size[1]
            self.input_w = previous_layer.size[2]
        self.output_c = self.filters_num
        self.output_h = (self.input_h - self.kernel_size) + 1
        self.output_w = (self.input_w - self.kernel_size) + 1
        self.z = torch.zeros((self.output_c, self.output_h, self.output_w))
        self.a = torch.zeros((self.output_c, self.output_h, self.output_w))
        self.w_shape = (self.output_c, self.input_c, self.kernel_size, self.kernel_size)
        self.w = self._init_weights(self.w_shape)
        self.b = super()._init_biases((self.output_c, 1))
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)

        return self

    def zero_grad(self):
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)
        return

    def _convolution(self, input, filters):
        tensor_filters = torch.zeros(
            self.output_c,
            self.output_h,
            self.output_w,
            self.input_c,
            self.kernel_size,
            self.kernel_size
        )
        for f in range(filters.size(0)):
            expanded_filter = filters[f].unsqueeze(0).unsqueeze(0)
            tensor_filters[f] = expanded_filter.expand(self.output_h, self.output_w, -1, -1, -1)

        regions = torch.zeros(
            self.output_h,
            self.output_w,
            self.input_c,
            self.kernel_size,
            self.kernel_size
        )
        k = self.kernel_size
        for i in range(self.output_h):
            for j in range(self.output_w):
                regions[i][j] = input[:, i:i + k, j:j + k]

        expanded_regions = regions.unsqueeze(0)
        tensor_regions = expanded_regions.expand(self.output_c, -1, -1, -1, -1, -1)

        return torch.sum(tensor_regions * tensor_filters, dim=(3, 4, 5))

    def forward(self, input):

        start_for_time = time.time()

        self.z = self._convolution(input, self.w)

        self.a = self.activation.apply(self.z)

        # print('Forward time: ', time.time() - start_for_time)
        #
        # print('self.a[0]')
        # print(self.a[0])
        #
        # time.sleep(1)

        return

    # def forward(self, input):
    #
    #     weights = torch.zeros(self.output_c, self.output_h, self.output_w, self.input_c, self.kernel_size, self.kernel_size)
    #     for f in range(self.filters_num):
    #
    #         # print('self.w[f]')
    #         # print(self.w[f])
    #
    #         expanded_tensor = self.w[f].unsqueeze(0).unsqueeze(0)
    #         weights[f] = expanded_tensor.expand(self.output_h, self.output_w, -1, -1, -1)
    #
    #     # print('weights')
    #     # print(weights.shape)
    #     # print(weights)
    #
    #     start_for_time = time.time()
    #
    #     k = self.kernel_size
    #
    #     # print('input')
    #     # print(input)
    #
    #     input_regions = torch.zeros(self.output_c, self.output_h, self.output_w, self.input_c, self.kernel_size, self.kernel_size)
    #     regions = torch.zeros(self.output_h, self.output_w, self.input_c, self.kernel_size, self.kernel_size)
    #     for i in range(self.output_h):
    #         for j in range(self.output_w):
    #             regions[i][j] = input[:, i:i + k, j:j + k]
    #
    #     expanded_regions = regions.unsqueeze(0)
    #     input_regions = expanded_regions.expand(self.output_c, -1, -1, -1, -1, -1)
    #
    #     # print()
    #     # print('input_regions')
    #     # print(input_regions.shape)
    #     # print(input_regions)
    #
    #     # for f in range(self.filters_num):
    #     #     self.z[f] = torch.sum(input_regions * weights[f], dim=(2,3,4))
    #     self.z = torch.sum(input_regions * weights, dim=(3,4,5))
    #
    #     self.a = self.activation.apply(self.z)
    #
    #     # print('Forward time: ', time.time() - start_for_time)
    #     #
    #     # print('self.a[0]')
    #     # print(self.a[0])
    #
    #     #time.sleep(1)
    #
    #     return

    # def forward(self, input):
    #
    #     start_for_time = time.time()
    #
    #     k = self.kernel_size
    #     for f in range(self.filters_num):
    #         for i in range(self.output_h):
    #             for j in range(self.output_w):
    #                 self.z[f][i][j] = (
    #                     torch.sum(input[:, i:i + k, j:j + k] * self.w[f])
    #                     + self.b[f]
    #                 )
    #
    #     self.a = self.activation.apply(self.z)
    #
    #     print('Forward time: ', time.time() - start_for_time)
    #
    #     print('self.a[0]')
    #     print(self.a[0])
    #
    #     time.sleep(1)
    #
    #     return

    def backward(self, next_layer_error, prev_layer, next_layer):
        layer_error = self._calculate_layer_error(next_layer_error, next_layer)
        for f in range(self.filters_num):
            layer_error_f = layer_error[f]
            for c in range(self.input_c):
                prev_layer_a_c = prev_layer.a[c]
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        self.grad_w[f][c][m][n] += torch.sum(
                            layer_error_f
                            * prev_layer_a_c[m:m + self.output_h, n:n + self.output_w]
                        )
            self.grad_b[f] += torch.sum(layer_error_f)

        return layer_error


class Flatten:
    def __init__(self):
        self.type = "flatten"
        self.learnable = False

    def initialize(self, previous_layer):
        if previous_layer.type != 'convolutional':
            raise Exception("Flatten layer should be used only after convolutional.")

        self.size = (previous_layer.output_c
                     * previous_layer.output_h
                     * previous_layer.output_w)
        self.a = torch.zeros((self.size, 1))

        return self

    def forward(self, input):
        self.a = input.flatten()
        self.a = self.a.reshape(self.a.size(0), 1)
        return

    def backward(self, layer_error, prev_layer, next_layer):
        reshape_sizes = (
            prev_layer.output_c,
            prev_layer.output_h,
            prev_layer.output_w
        )
        layer_error = (torch.matmul(next_layer.w.t(), layer_error)
                       * torch.ones_like(self.a)).reshape(reshape_sizes)

        return layer_error