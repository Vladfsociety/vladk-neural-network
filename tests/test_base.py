import pytest
import time
import torch

from src.model.base import NeuralNetwork
from src.model.layer import Input, FullyConnected
from src.model.loss import MeanSquaredError
from src.model.optimizer import SGD
from src.model.metric import R2Score
from src.model.activation import Relu, Linear

"""
Regression testing on 2D functions
"""
def func_quadratic(x):
    return 0.5 * x ** 2 + 2 * x - 1

def func_linear(x):
    return -x - 1

def get_dataset_regression(func):
    def create_data(x_values):
        return [{'input': [x_i], 'output': [func(x_i)]} for x_i in x_values]

    train_data = create_data(torch.linspace(-5, 5, 100))
    test_data = create_data(torch.linspace(-5, 5, 30))

    return train_data, test_data

def run_regression_test(func, learning_rate=0.001, mse_threshold=0.1, r2_threshold=0.975, fit_time_threshold=2.0):
    train_dataset, test_dataset = get_dataset_regression(func)

    layers = [
        FullyConnected(128, Relu()),
        FullyConnected(128, Relu()),
        FullyConnected(128, Relu()),
        FullyConnected(1, Linear())
    ]
    nn = NeuralNetwork(
        Input(1),
        layers,
        optimizer=SGD(learning_rate=learning_rate),
        loss=MeanSquaredError(),
        metric=R2Score(),
    )

    start_time = time.time()
    epochs = 50
    nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=1, verbose=True)
    fit_time = round(time.time() - start_time, 4)

    prediction = nn.predict(test_dataset)
    actual = torch.tensor([test_sample['output'] for test_sample in test_dataset])

    mse = nn.loss(prediction, actual)
    r2 = nn.metric(prediction, actual)

    print(f"MSE: {mse}, r2: {r2}, Fit time: {fit_time}")

    assert mse < mse_threshold
    assert r2 > r2_threshold
    assert fit_time < fit_time_threshold

@pytest.mark.two_dim
@pytest.mark.parametrize("func", [
    (func_quadratic),
    (func_linear)
])
def test_regression(func):
    run_regression_test(func)

"""
Regression testing on 3D functions
"""
def func_quadratic_3d(x, y):
    return 0.2 * x ** 2 + 0.2 * y ** 2

def func_sin_plus_cos_3d(x, y):
    return torch.sin(x) + torch.cos(y)

def get_dataset_regression_3d(func):
    def create_data(x_values, y_values):
        data = []
        for i in range(x_values.size(0)):
            for j in range(x_values.size(1)):
                data.append({
                    'input': [x_values[i][j], y_values[i][j]],
                    'output': [func(x_values[i][j], y_values[i][j])],
                })
        return data

    x_train = torch.linspace(-5, 5, 50)
    y_train = torch.linspace(-5, 5, 50)
    x_train, y_train = torch.meshgrid(x_train, y_train, indexing='ij')

    train_data = create_data(x_train, y_train)

    x_test = torch.linspace(-5, 5, 30)
    y_test = torch.linspace(-5, 5, 30)
    x_test, y_test = torch.meshgrid(x_test, y_test, indexing='ij')

    test_data = create_data(x_test, y_test)

    return train_data, test_data

def run_regression_test_3d(func, learning_rate=0.001, mse_threshold=0.05, r2_threshold=0.975, fit_time_threshold=40.0):
    train_dataset, test_dataset = get_dataset_regression_3d(func)

    layers = [
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(1, Linear())
    ]
    nn = NeuralNetwork(
        Input(2),
        layers,
        optimizer=SGD(learning_rate=learning_rate),
        loss=MeanSquaredError(),
        metric=R2Score(),
    )

    start_time = time.time()
    epochs = 20
    nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=16, verbose=True)
    fit_time = round(time.time() - start_time, 4)

    prediction = nn.predict(test_dataset)
    actual = torch.tensor([test_sample['output'] for test_sample in test_dataset])

    mse = nn.loss(prediction, actual)
    r2 = nn.metric(prediction, actual)

    print(f"MSE: {mse}, r2: {r2}, Fit time: {fit_time}")

    assert mse < mse_threshold
    assert r2 > r2_threshold
    assert fit_time < fit_time_threshold

@pytest.mark.three_dim
@pytest.mark.parametrize("func, learning_rate", [
    (func_quadratic_3d, 0.001),
    (func_sin_plus_cos_3d, 0.025)
])
def test_regression_3d(func, learning_rate):
    run_regression_test_3d(func, learning_rate)

