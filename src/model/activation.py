import torch
import torch.nn.functional as f


class Relu:
    def apply(self, x):
        return f.relu(x)

    def derivative(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


class LeakyRelu:
    def __init__(self, negative_slope=0.01):
        self.__negative_slope = negative_slope

    def apply(self, x):
        return f.leaky_relu(x, negative_slope=self.__negative_slope)

    def derivative(self, x):
        return torch.where(
            x > 0, torch.ones_like(x), torch.full_like(x, self.__negative_slope)
        )


class Linear:
    def apply(self, x):
        return x

    def derivative(self, x):
        return torch.ones_like(x)


class Sigmoid:
    def __init__(self, epsilon=1e-06, min_arg_value=-100, max_arg_value=100):
        self.__epsilon = epsilon
        self.__min_arg_value = min_arg_value
        self.__max_arg_value = max_arg_value

    def apply(self, x):
        x = torch.clamp(x, min=self.__min_arg_value, max=self.__max_arg_value)
        return f.sigmoid(x)

    def derivative(self, x):
        value = self.apply(x)
        derivative = value * (torch.ones_like(value) - value)
        return torch.clamp(derivative, min=self.__epsilon, max=1.0 - self.__epsilon)
