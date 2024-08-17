import torch


class MeanSquaredError:
    def value(self, prediction, actual):
        return 0.5 * ((actual - prediction) ** 2).sum() / prediction.size(0)

    def derivative(self, prediction, actual):
        return prediction - actual

class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-10):
        self.__epsilon = epsilon

    def value(self, prediction, actual):

        prediction = torch.clamp(prediction, min=self.__epsilon, max=1 - self.__epsilon )

        losses = -(actual * torch.log(prediction) + (torch.ones_like(prediction) - actual) *
                   torch.log(torch.ones_like(prediction) - prediction))

        return losses.sum() / prediction.size(0)

    def derivative(self, prediction, actual):

        prediction = torch.clamp(prediction, min=self.__epsilon, max=1 - self.__epsilon)

        return (prediction - actual) / ((prediction * (1 - prediction)) + self.__epsilon)