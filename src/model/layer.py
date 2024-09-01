import math
import random

import torch

from src.model.activation import Linear

class Layer:
    def __init__(self, size):
        self.size = size

    # def _init_weights(self, size):
    #     fan_in, fan_out = size[0], size[1]
    #     limit = math.sqrt(6 / (fan_in + fan_out))
    #     return torch.tensor(
    #         [
    #             [random.uniform(-limit, limit) for _ in range(size[1])]
    #             for _ in range(size[0])
    #         ]
    #     )

    def _init_weights(self, size):
        fan_in, fan_out = size[0], size[1]
        limit = math.sqrt(1 / fan_in)
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
        return {
            "type": 'input',
            "a": torch.zeros((self.size, 1))
        }


class FullyConnected(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.__activation = activation

    def initialize(self, previous_layer):
        previous_layer_size = previous_layer.size

        print('previous_layer_size')
        print(previous_layer_size)

        return {
            "type": 'fully_connected',
            "learnable": True,
            "w": super()._init_weights((self.size, previous_layer_size)),
            "b": super()._init_biases((self.size, 1)),
            "z": torch.zeros((self.size, 1)),
            "activation_function": self.__activation,
            "a": torch.zeros((self.size, 1)),
        }


class Input3D(Layer):
    def initialize(self):
        print('self.sizeInput3D(Layer)')
        print(self.size)
        return {
            "type": 'input_3d',
            "a": torch.zeros((self.size[0], self.size[1], self.size[2]))
        }


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
    def __init__(self, activation, filters=2, kernel=2, padding=0, stride=1):
        size = (kernel, kernel, 1)
        super().__init__(size)
        self.__activation = activation
        self.__kernel = kernel
        self.__padding = padding
        self.__stride = stride
        self.__filters = filters

    def initialize(self, previous_layer):
        self.__input_c = previous_layer.size[0]
        self.__input_h = previous_layer.size[1]
        self.__input_w = previous_layer.size[2]

        print('previous_layer.size')
        print(previous_layer.size)

        self.__output_c = self.__filters
        self.__output_h = (self.__input_h - self.__kernel) + 1
        self.__output_w = (self.__input_w - self.__kernel) + 1
        self.size = (self.__output_c, self.__output_h, self.__output_w)
        return {
            "type": 'convolutional',
            "learnable": True,
            "w": self._init_weights((self.__output_c, self.__input_c, self.__kernel, self.__kernel)),
            "b": super()._init_biases((self.__output_c, 1)),
            #"z": torch.zeros((self.__output_c, self.__output_h, self.__output_w, 1)),
            "z": torch.zeros((self.__output_c, self.__output_h, self.__output_w)),
            "activation_function": self.__activation,
            #"a": torch.zeros((self.__output_c, self.__output_h, self.__output_w, 1)),
            "a": torch.zeros((self.__output_c, self.__output_h, self.__output_w)),
            "kernel": self.__kernel,
            "filters_num": self.__filters,
            "input_h": self.__input_h,
            "input_w": self.__input_w,
            "input_c": self.__input_c,
            "output_h": self.__output_h,
            "output_w": self.__output_w,
            "output_c": self.__output_c,
            "w_shape": (self.__output_c, self.__input_c, self.__kernel, self.__kernel),
        }

    def _init_weights(self, size):

        print('size')
        print(size)

        n_inputs = size[1] * size[2] * size[3]

        print('n_inputs')
        print(n_inputs)

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


class Flatten:
    def __init__(self):
        self.size = None

    def initialize(self, previous_layer):
        #layer_size = previous_layer.numel()

        print('previous_layer.size')
        print(previous_layer.size)

        self.size = previous_layer.size[0] * previous_layer.size[1] * previous_layer.size[2]
        return {
            "type": 'flatten',
            "learnable": False,
            "w": torch.zeros((self.size, 1)),
            "b": torch.zeros((self.size, 1)),
            "z": torch.zeros((self.size, 1)),
            #"activation_function": self.__activation,
            "a": torch.zeros((self.size, 1)),
        }