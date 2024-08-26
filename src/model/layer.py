import math
import random

import torch

from src.model.activation import Linear

class Layer:
    def __init__(self, size):
        self.size = size

    def _init_weights(self, size):
        fan_in, fan_out = size[0], size[1]
        limit = math.sqrt(6 / (fan_in + fan_out))
        return torch.tensor(
            [
                [random.uniform(-limit, limit) for _ in range(size[1])]
                for _ in range(size[0])
            ]
        )

    def _init_biases(self, size):
        # fan_in, fan_out = size[0], size[1]
        # limit = math.sqrt(6 / (fan_in + fan_out))
        # return torch.tensor(
        #     [
        #         [random.uniform(-limit, limit) for _ in range(size[1])]
        #         for _ in range(size[0])
        #     ]
        # )
        return torch.zeros(size)


class Input(Layer):
    def initialize(self):
        return {"a": torch.zeros((self.size, 1))}


class FullyConnected(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.__activation = activation

    def initialize(self, previous_layer):
        previous_layer_size = previous_layer.size

        print('previous_layer_size')
        print(previous_layer_size)

        return {
            "w": super()._init_weights((self.size, previous_layer_size)),
            "b": super()._init_biases((self.size, 1)),
            "z": torch.zeros((self.size, 1)),
            "activation_function": self.__activation,
            "a": torch.zeros((self.size, 1)),
        }


class Input3D(Layer):
    def initialize(self):
        return {"a": torch.zeros((self.size[0], self.size[1], self.size[2]))}


# class Convolutional(Layer):
#     def __init__(self, filters, activation, kernel_size, padding, stride):
#         size = (kernel_size, kernel_size, filters)
#         super().__init__(size)
#         self.__activation = activation
#         self.__kernel_size = kernel_size
#         self.__padding = padding
#         self.__stride = stride
#
#     def initialize(self, previous_layer_size):
#         return {
#             "w": super()._init_weights((self.size, previous_layer_size)),
#             "b": super()._init_biases((self.size, 1)),
#             "z": torch.zeros((self.size[0], self.size[1], self.size[2])),
#             "activation_function": self.__activation,
#             "a": torch.zeros((self.size[0], self.size[1], self.size[2])),
#         }

class Convolutional3x3x16x0x1(Layer):
    def __init__(self, activation, filters=1, kernel_size=3, padding=0, stride=1):
        size = (kernel_size, kernel_size, 1)
        super().__init__(size)
        self.__activation = activation
        self.__kernel_size = kernel_size
        self.__padding = padding
        self.__stride = stride

    def initialize(self, previous_layer):
        res_1 = (previous_layer.size[0] - 3) + 1
        res_2 = (previous_layer.size[1] - 3) + 1
        res_3 = 1
        self.size = (res_3, res_1, res_2)
        return {
            "w": self._init_weights((1, 3, 3, 1)),
            "b": super()._init_biases((1, 1)),
            "z": torch.zeros((res_1, res_2, res_3)),
            "activation_function": self.__activation,
            "a": torch.zeros((res_1, res_2, res_3)),
        }

    def _init_weights(self, size):
        n_inputs = size[0] * size[1] * size[2]

        print('n_inputs')
        print(n_inputs)

        std = math.sqrt(2 / n_inputs)
        return torch.tensor(
            [
                [[[random.gauss(0, std) for _ in range(size[3])]
                for _ in range(size[2])]
                for _ in range(size[1])]
                for _ in range(size[0])
            ]
        )


class Flatten:
    def __init__(self):
        self.size = None

    def initialize(self, previous_layer):
        #layer_size = previous_layer.numel()

        print('previous_layer.size')
        print(previous_layer.size)

        self.size = previous_layer.size[0]*previous_layer.size[1]*previous_layer.size[2]
        return {
            "w": torch.zeros((self.size, 1)),
            "b": torch.zeros((self.size, 1)),
            "z": torch.zeros((self.size, 1)),
            #"activation_function": self.__activation,
            "a": torch.zeros((self.size, 1)),
        }