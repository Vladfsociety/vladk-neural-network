import random
import torch
import math


class Layer:
    def __init__(self, size):
        self._size = size

    @property
    def size(self):
        return self._size

    def _init_weights(self, size):
        fan_in, fan_out = size[0], size[1]
        limit = math.sqrt(6 / (fan_in + fan_out))
        return torch.tensor([[random.uniform(-limit, limit) for _ in range(size[1])] for _ in range(size[0])])
        #return torch.tensor([[random.uniform(-0.1, 0.1) for _ in range(size[1])] for _ in range(size[0])])

    def _init_biases(self, size):
        fan_in, fan_out = size[0], size[1]
        limit = math.sqrt(6 / (fan_in + fan_out))
        return torch.tensor([[random.uniform(-limit, limit) for _ in range(size[1])] for _ in range(size[0])])
        #return torch.tensor([[random.uniform(-0.1, 0.1) for _ in range(size[1])] for _ in range(size[0])])

class Input(Layer):
    def initialize(self):
        return {
            'a': torch.zeros((self._size, 1))
        }

class FullyConnected(Layer):
    def __init__(self, size, activation):
        super().__init__(size)
        self.__activation = activation

    def initialize(self, previous_layer_size):
        return {
            'w': super()._init_weights((self._size, previous_layer_size)),
            'b': super()._init_biases((self._size, 1)),
            'z': torch.zeros((self._size, 1)),
            'activation_function': self.__activation,
            'a': torch.zeros((self._size, 1))
        }
