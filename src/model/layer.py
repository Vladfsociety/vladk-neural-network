import math
import random

import torch


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


    def initialize(self, previous_layer):
        previous_layer_size = previous_layer.size
        self.z = torch.zeros((self.size, 1))
        self.a = torch.zeros((self.size, 1))
        self.w = super()._init_weights((self.size, previous_layer_size))
        self.b = super()._init_biases((self.size, 1))
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)

    def zero_grad(self):
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)
        return

    def forward(self, input):
        self.z = torch.matmul(self.w, input) + self.b
        self.a = self.activation.apply(self.z)
        return

    def backward(self, next_layer_error, next_layer=None):

        layer_error = self._calculate_layer_error(next_layer_error, next_layer)

        self.grad_w += torch.matmul(layer_error, self.a.t())
        self.grad_b += layer_error

        return layer_error



    def _calculate_layer_error(self, next_layer_error, next_layer):

        if not next_layer:
            layer_error = next_layer_error * self.activation.derivative(self.z)
        else:
            layer_error = (torch.matmul(next_layer.w.t(), next_layer_error)
                           * self.activation.derivative(self.z))

        return layer_error

class Convolutional(Layer):
    def __init__(self, activation, filters_num=2, kernel_size=3, padding=0, stride=1):
        self.type = "convolutional"
        self.learnable = True
        self.activation = activation
        self.filters_num = filters_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def initialize(self, previous_layer):
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

    def zero_grad(self):
        self.grad_w = torch.zeros_like(self.w)
        self.grad_b = torch.zeros_like(self.b)
        return

    def forward(self, input):

        k = self.kernel_size

        for f in range(self.filters_num):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    self.z[f][i][j] = (
                        torch.sum(input[:, i:i + k, j:j + k] * self.w[f])
                        + self.b[f]
                    )

        self.a = self.activation.apply(self.z)
        return

    def backward(self, next_layer_error, next_layer):

        layer_error = self._calculate_layer_error(next_layer_error, next_layer)

        for f in range(self.filters_num):
            for c in range(self.input_c):
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        self.grad_w[f][c][m][n] = torch.sum(
                            layer_error[f]
                            * self.a[c, m:m + self.output_h:, n:n + self.output_w:]
                        )

            self.grad_b[f] = torch.sum(layer_error[f])

        return layer_error

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

            layer_error_with_padding = self._get_padded_tensor(next_layer_error, (1, 1, 1, 1))

            layer_error = torch.zeros((self.output_c, self.output_h, self.output_w))

            flipped_w_next = torch.flip(next_layer.w, (2, 3))

            k_next = next_layer.kernel_size

            for f in range(self.filters_num):
                for i in range(self.output_h):
                    for j in range(self.output_w):
                        layer_error[f][i][j] = (
                            torch.sum(
                                layer_error_with_padding[:, i:i + k_next, j:j + k_next]
                                * flipped_w_next[:, f]
                            )
                            * self.activation.derivative(self.z[f][i][j])
                        )

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

# class Convolutional3x3x16x0x1(Layer):
#     def __init__(self, activation, filters=2, kernel=3, padding=0, stride=1):
#         size = (kernel, kernel, 1)
#         super().__init__(size)
#         self.__activation = activation
#         self.__kernel = kernel
#         self.__padding = padding
#         self.__stride = stride
#         self.__filters = filters
#
#     def initialize(self, previous_layer, device):
#
#         self.__device = device
#
#         self.__input_c = previous_layer.size[0]
#         self.__input_h = previous_layer.size[1]
#         self.__input_w = previous_layer.size[2]
#
#         # print('previous_layer.size')
#         # print(previous_layer.size)
#
#         self.__output_c = self.__filters
#         self.__output_h = (self.__input_h - self.__kernel) + 1
#         self.__output_w = (self.__input_w - self.__kernel) + 1
#         self.size = (self.__output_c, self.__output_h, self.__output_w)
#         return {
#             "type": 'convolutional',
#             "learnable": True,
#             "w": self._init_weights((self.__output_c, self.__input_c, self.__kernel, self.__kernel), device),
#             "b": super()._init_biases((self.__output_c, 1), device),
#             #"z": torch.zeros((self.__output_c, self.__output_h, self.__output_w, 1)),
#             "z": torch.zeros((self.__output_c, self.__output_h, self.__output_w), device=self.__device),
#             "activation_function": self.__activation,
#             #"a": torch.zeros((self.__output_c, self.__output_h, self.__output_w, 1)),
#             "a": torch.zeros((self.__output_c, self.__output_h, self.__output_w), device=self.__device),
#             "kernel": self.__kernel,
#             "filters_num": self.__filters,
#             "input_h": self.__input_h,
#             "input_w": self.__input_w,
#             "input_c": self.__input_c,
#             "output_h": self.__output_h,
#             "output_w": self.__output_w,
#             "output_c": self.__output_c,
#             "w_shape": (self.__output_c, self.__input_c, self.__kernel, self.__kernel),
#         }
#
#     def _init_weights(self, size, device):
#
#         # print('size')
#         # print(size)
#
#         n_inputs = size[1] * size[2] * size[3]
#
#         # print('n_inputs')
#         # print(n_inputs)
#
#         random.seed(41)
#
#         limit = math.sqrt(1 / n_inputs)
#         return torch.tensor(
#             [
#                 [[[random.uniform(-limit, limit) for _ in range(size[3])]
#                 for _ in range(size[2])]
#                 for _ in range(size[1])]
#                 for _ in range(size[0])
#             ],
#             device=device
#         )


class Flatten:
    def __init__(self):
        self.type = "flatten"
        self.learnable = False

    def initialize(self, previous_layer):

        if previous_layer.type is not 'convolutional':
            raise Exception("Flatten layer should be used only after convolutional.")

        self.size = (previous_layer.output_c
                     * previous_layer.output_h
                     * previous_layer.output_w)
        self.a = torch.zeros((self.size, 1))

    def forward(self, input):
        self.a = input.flatten()
        self.a = self.a.reshape(self.a.size(0), 1)
        return

    def backward(self, layer_error, next_layer):

        reshape_sizes = (
            self.size[0],
            self.size[1],
            self.size[2],
        )

        layer_error = (torch.matmul(next_layer.w.t(), layer_error)
                       * torch.ones_like(self.a)).reshape(reshape_sizes)

        return layer_error