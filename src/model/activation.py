import torch
import torch.nn.functional as f


class Relu:
    def apply(self, argument):
        return f.relu(argument)

    def derivative(self, argument):
        return torch.where(argument > 0, torch.ones_like(argument), torch.zeros_like(argument))

class LeakyRelu:
    def __init__(self, negative_slope=0.01):
        self.__negative_slope = negative_slope

    def apply(self, argument):
        return f.leaky_relu(argument, negative_slope=self.__negative_slope)

    def derivative(self, argument):
        return torch.where(argument > 0, torch.ones_like(argument), torch.full_like(argument, self.__negative_slope))

class Linear:
    def apply(self, argument):
        return argument

    def derivative(self, argument):
        return torch.ones_like(argument)

class Sigmoid:
    def __init__(self, epsilon=1e-12, min_arg_value=-100, max_arg_value=100):
        self.__epsilon       = epsilon
        self.__min_arg_value = min_arg_value
        self.__max_arg_value = max_arg_value

    def apply(self, argument):

        argument = torch.clamp(argument, min=self.__min_arg_value, max=self.__max_arg_value)

        return f.sigmoid(argument)

    def derivative(self, argument):

        value = self.apply(argument)

        derivative = value * (torch.ones_like(value) - value)

        return torch.clamp(derivative, min=self.__epsilon, max=1-self.__epsilon)
        #return argument * (torch.ones_like(argument) - argument)