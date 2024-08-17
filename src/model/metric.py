import pprint

import torch

class R2Score:
    def name(self):
        return 'R2 score'

    def value(self, prediction, actual):
        prediction = torch.flatten(torch.tensor(prediction))
        actual = torch.flatten(torch.tensor(actual))

        ss_res = torch.sum((actual - prediction) ** 2)
        ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)

        return 1.0 - (ss_res / ss_tot)

class Accuracy:
    def name(self):
        return 'Accuracy'

    def value(self, prediction, actual):

        prediction = torch.flatten(torch.tensor(prediction))
        actual = torch.flatten(torch.tensor(actual))

        # print('prediction')
        # pprint.pprint(prediction)
        # print('actual')
        # pprint.pprint(actual)
        # print('prediction == actual')
        # pprint.pprint(prediction == actual)
        # print('(prediction == actual).sum().item() / prediction.size(0)')
        # pprint.pprint((prediction == actual).sum().item() / prediction.size(0))

        return (prediction == actual).sum().item() / prediction.size(0)