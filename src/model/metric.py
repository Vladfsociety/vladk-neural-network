import torch


def r2_score(prediction, actual):

    prediction = torch.flatten(torch.tensor(prediction))
    actual = torch.flatten(torch.tensor(actual))

    ss_res = torch.sum((actual - prediction) ** 2)
    ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)

    return 1.0 - (ss_res / ss_tot)

def mse_loss(prediction, actual):

    prediction = torch.flatten(torch.tensor(prediction))
    actual = torch.flatten(torch.tensor(actual))

    return ((actual - prediction)**2).sum()/prediction.size(0)
