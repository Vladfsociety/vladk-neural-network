import math
import random

import torch


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
        fan_in, fan_out = size[0], size[1]
        limit = math.sqrt(6 / (fan_in + fan_out))
        return torch.tensor(
            [
                [random.uniform(-limit, limit) for _ in range(size[1])]
                for _ in range(size[0])
            ]
        )


class Input(Layer):
    def initialize(self):
        return {"a": torch.zeros((self.size, 1))}


class FullyConnected(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.__activation = activation

    def initialize(self, previous_layer_size):
        return {
            "w": super()._init_weights((self.size, previous_layer_size)),
            "b": super()._init_biases((self.size, 1)),
            "z": torch.zeros((self.size, 1)),
            "activation_function": self.__activation,
            "a": torch.zeros((self.size, 1)),
        }
