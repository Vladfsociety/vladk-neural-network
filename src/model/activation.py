import torch
import torch.nn.functional as f


class Relu:
    def apply(self, argument):
        return f.relu(argument)

    def derivative(self, argument):
        return torch.where(argument > 0, torch.ones_like(argument), torch.zeros_like(argument))

class Linear:
    def apply(self, argument):
        return argument

    def derivative(self, argument):
        return torch.ones_like(argument)

class Sigmoid:
    def apply(self, argument):
        return f.sigmoid(argument)

    def derivative(self, argument):
        return argument * (torch.ones_like(argument) - argument)