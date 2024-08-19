import pprint

import torch


class MeanSquaredError:
    def value(self, prediction, actual):

        # print('MeanSquaredError_prediction')
        # print(prediction)
        # print('MeanSquaredError_actual')
        # print(actual)

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


class CategoricalCrossEntropy:
    # def __init__(self, epsilon=1e-10):
    #     self.__epsilon = epsilon

    def value(self, prediction, actual):

        # print('CategoricalCrossEntropy_value')
        # print('prediction')
        # pprint.pprint(prediction)
        # print('actual')
        # pprint.pprint(actual)

        softmax_prediction = torch.softmax(prediction, dim=1)

        # print('softmax_prediction')
        # pprint.pprint(softmax_prediction)

        #prediction = torch.clamp(prediction, min=self.__epsilon, max=1 - self.__epsilon )

        losses = -(actual * torch.log(softmax_prediction))

        # print('losses')
        # pprint.pprint(losses)

        return losses.sum() / prediction.size(0)

    def derivative(self, prediction, actual):

        # print('CategoricalCrossEntropy_derivative')
        # print('prediction')
        # pprint.pprint(prediction)
        # print('actual')
        # pprint.pprint(actual)
        # print('result')
        # pprint.pprint(prediction - actual)

        return prediction - actual