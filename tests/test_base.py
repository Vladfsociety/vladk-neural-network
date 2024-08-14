import pytest
import time
import torch
import torch.nn.functional as f
from src.model.base import NeuralNetwork
from src.model.layer import FullyConnected
from src.model.loss import MeanSquaredError
from src.model.opimizer import SGD
from src.model.metric import r2_score, mse_loss


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

def run_regression_test(func, mse_threshold=0.1, r2_threshold=0.95, fit_time_threshold=5.0):
    train_dataset, test_dataset = get_dataset_regression(func)

    layers = [
        FullyConnected(128, f.relu),
        FullyConnected(128, f.relu),
        FullyConnected(128, f.relu),
        FullyConnected(1, 'linear')
    ]
    nn = NeuralNetwork(1, layers, optimizer=SGD(), loss=MeanSquaredError())

    start_time = time.time()
    epochs = 100
    nn.fit(train_dataset, test_dataset, epochs=epochs)
    fit_time = time.time() - start_time

    prediction = nn.predict(test_dataset)
    actual = [test_sample['output'] for test_sample in test_dataset]

    mse = mse_loss(prediction, actual)
    r2 = r2_score(prediction, actual)

    print(f"MSE: {mse}, r2: {r2}, Fit time: {fit_time}")

    assert mse < mse_threshold
    assert r2 > r2_threshold
    assert fit_time < fit_time_threshold

@pytest.mark.two_dim
@pytest.mark.parametrize("func", [func_quadratic, func_linear])
def test_regression(func):
    if func.__name__ == "func_quadratic":
        run_regression_test(func, mse_threshold=0.7)
    else:
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

    x_train = torch.linspace(-5, 5, 100)
    y_train = torch.linspace(-5, 5, 100)
    x_train, y_train = torch.meshgrid(x_train, y_train, indexing='ij')

    train_data = create_data(x_train, y_train)

    x_test = torch.linspace(-5, 5, 30)
    y_test = torch.linspace(-5, 5, 30)
    x_test, y_test = torch.meshgrid(x_test, y_test, indexing='ij')

    test_data = create_data(x_test, y_test)

    return train_data, test_data

def run_regression_test_3d(func, mse_threshold=0.2, r2_threshold=0.95, fit_time_threshold=450.0):
    train_dataset, test_dataset = get_dataset_regression_3d(func)

    layers = [
        FullyConnected(256, f.relu),
        FullyConnected(256, f.relu),
        FullyConnected(256, f.relu),
        FullyConnected(256, f.relu),
        FullyConnected(256, f.relu),
        FullyConnected(1, 'linear')
    ]
    nn = NeuralNetwork(2, layers, optimizer=SGD(), loss=MeanSquaredError())

    start_time = time.time()
    epochs = 80
    nn.fit(train_dataset, test_dataset, epochs=epochs)
    fit_time = time.time() - start_time

    prediction = nn.predict(test_dataset)
    actual = [test_sample['output'] for test_sample in test_dataset]

    mse = mse_loss(prediction, actual)
    r2 = r2_score(prediction, actual)

    print(f"MSE: {mse}, r2: {r2}, Fit time: {fit_time}")

    assert mse < mse_threshold
    assert r2 > r2_threshold
    assert fit_time < fit_time_threshold

@pytest.mark.three_dim
@pytest.mark.parametrize("func", [func_quadratic_3d, func_sin_plus_cos_3d])
def test_regression_3d(func):
    if func.__name__ == "func_sin_plus_cos_3d":
        run_regression_test_3d(func, r2_threshold=0.9)
    else:
        run_regression_test_3d(func)

