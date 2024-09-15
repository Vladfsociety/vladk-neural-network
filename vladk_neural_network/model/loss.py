import torch


class MeanSquaredError:
    def value(self, prediction, actual):
        return 0.5 * ((actual - prediction) ** 2).sum() / prediction.size(0)

    def derivative(self, prediction, actual):
        return prediction - actual


class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-06):
        self.__epsilon = epsilon

    def value(self, prediction, actual):
        prediction = torch.clamp(
            prediction, min=self.__epsilon, max=1.0 - self.__epsilon
        )
        losses = -(
            actual * torch.log(prediction) + (1 - actual) * torch.log(1 - prediction)
        )
        return losses.sum() / prediction.size(0)

    def derivative(self, prediction, actual):
        prediction = torch.clamp(
            prediction, min=self.__epsilon, max=1.0 - self.__epsilon
        )
        return (prediction - actual) / (
            (prediction * (1 - prediction)) + self.__epsilon
        )


class CategoricalCrossEntropy:
    def __init__(self, epsilon=1e-06):
        self.__epsilon = epsilon

    def value(self, prediction, actual):
        log_softmax_prediction = torch.log_softmax(prediction, dim=1)
        losses = -(actual * log_softmax_prediction)
        return losses.sum() / prediction.size(0)

    def derivative(self, prediction, actual):
        softmax_prediction = torch.softmax(prediction, dim=0)
        return softmax_prediction - actual
