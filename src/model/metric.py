import torch


class R2Score:
    def name(self):
        return "R2 score"

    def value(self, prediction, actual):
        ss_res = torch.sum((actual - prediction) ** 2)
        ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)

        return 1.0 - (ss_res / ss_tot)


class Accuracy:
    def name(self):
        return "Accuracy"

    def value(self, prediction, actual):
        return (prediction == actual).sum().item() / prediction.size(0)


class AccuracyOneHot:
    def name(self):
        return "Accuracy"

    def value(self, prediction, actual):
        return (prediction * actual).to(torch.int).sum().item() / prediction.size(0)
