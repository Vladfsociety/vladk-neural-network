import torch.nn.functional as f
from model.base import NeuralNetwork
from model.layers import FullyConnected
from model.loss import MeanSquaredError
from model.opimizer import SGD

train_dataset = [
    {
        'input': [0.01, 0.01],
        'output': [0.01]
    },
    {
        'input': [0.1, 0.12],
        'output': [0.1]
    },
    {
        'input': [0.46, 0.51],
        'output': [0.5]
    },
    {
        'input': [0.6, 0.7],
        'output': [0.65]
    },
    {
        'input': [0.91, 0.92],
        'output': [0.92]
    },
]

layers = [
    FullyConnected(2, f.relu),
    FullyConnected(1, 'linear')
]
nn = NeuralNetwork(2, layers, optimizer=SGD(), loss=MeanSquaredError())

epochs = 1
nn.fit(train_dataset, epochs)
