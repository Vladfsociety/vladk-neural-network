import time

import pytest
import torch

from src.model.activation import Linear, Relu
from src.model.base import NeuralNetwork
from src.model.layer import FullyConnected, Input
from src.model.loss import MeanSquaredError
from src.model.metric import R2Score
from src.model.optimizer import SGD, Adam

"""
Regression testing on 2D functions.
"""


def func_quadratic(x):
    """Quadratic function for regression."""
    return 0.5 * x**2 + 2 * x - 1


def func_linear(x):
    """Linear function for regression."""
    return -x - 1


def get_dataset_regression(func):
    """Generate a dataset for 2D regression."""

    def create_data(x_values):
        return [{"input": [x_i], "output": [func(x_i)]} for x_i in x_values]

    train_data = create_data(torch.linspace(-5, 5, 100))
    test_data = create_data(torch.linspace(-5, 5, 30))

    return train_data, test_data


def run_regression_test(
    func,
    learning_rate=0.001,
    mse_threshold=0.1,
    r2_threshold=0.975,
    fit_time_threshold=5.0,
):
    """Run a regression test on a 2D function."""
    print(f"\nRegression. Testing {func.__name__}")

    train_dataset, test_dataset = get_dataset_regression(func)

    layers = [
        FullyConnected(128, Relu()),
        FullyConnected(128, Relu()),
        FullyConnected(128, Relu()),
        FullyConnected(1, Linear()),
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
    nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=1, verbose=False)
    fit_time = round(time.time() - start_time, 4)

    prediction = nn.predict(test_dataset)
    actual = torch.stack(
        [
            torch.tensor(test_sample["output"]).unsqueeze(1)
            for test_sample in test_dataset
        ]
    )

    mse = nn.loss(prediction, actual)
    r2 = nn.metric(prediction, actual)

    print(f"MSE: {mse}, R2: {r2}, Fit time: {fit_time} seconds")

    assert mse < mse_threshold
    assert r2 > r2_threshold
    assert fit_time < fit_time_threshold


@pytest.mark.regression_two_dim
@pytest.mark.parametrize(
    "func, mse_threshold", [(func_quadratic, 0.2), (func_linear, 0.1)]
)
def test_regression(func, mse_threshold):
    """Test function for regression on 2D functions."""
    run_regression_test(func, mse_threshold=mse_threshold)
    return


"""
Regression testing on 3D functions.
"""


def func_quadratic_3d(x, y):
    """Quadratic function for 3D regression."""
    return 0.2 * x**2 + 0.2 * y**2


def func_sin_plus_cos_3d(x, y):
    """Sin plus cos function for 3D regression."""
    return torch.sin(x) + torch.cos(y)


def get_dataset_regression_3d(func):
    """Generate a dataset for 3D regression."""

    def create_data(x_values, y_values):
        data = []
        for i in range(x_values.size(0)):
            for j in range(x_values.size(1)):
                data.append(
                    {
                        "input": [x_values[i][j], y_values[i][j]],
                        "output": [func(x_values[i][j], y_values[i][j])],
                    }
                )
        return data

    x_train = torch.linspace(-5, 5, 50)
    y_train = torch.linspace(-5, 5, 50)
    x_train, y_train = torch.meshgrid(x_train, y_train, indexing="ij")

    train_data = create_data(x_train, y_train)

    x_test = torch.linspace(-5, 5, 30)
    y_test = torch.linspace(-5, 5, 30)
    x_test, y_test = torch.meshgrid(x_test, y_test, indexing="ij")

    test_data = create_data(x_test, y_test)

    return train_data, test_data


def run_regression_test_3d(
    func,
    learning_rate=0.001,
    mse_threshold=0.05,
    r2_threshold=0.975,
    fit_time_threshold=40.0,
):
    """Run a regression test on a 3D function."""
    print(f"\nRegression. Testing {func.__name__}")

    train_dataset, test_dataset = get_dataset_regression_3d(func)

    layers = [
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(256, Relu()),
        FullyConnected(1, Linear()),
    ]
    nn = NeuralNetwork(
        Input(2),
        layers,
        optimizer=Adam(learning_rate=learning_rate),
        loss=MeanSquaredError(),
        metric=R2Score(),
    )

    start_time = time.time()
    epochs = 15
    nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=16, verbose=False)
    fit_time = round(time.time() - start_time, 4)

    prediction = nn.predict(test_dataset)
    actual = torch.stack(
        [
            torch.tensor(test_sample["output"]).unsqueeze(1)
            for test_sample in test_dataset
        ]
    )

    mse = nn.loss(prediction, actual)
    r2 = nn.metric(prediction, actual)

    print(f"MSE: {mse}, R2: {r2}, Fit time: {fit_time} seconds")

    assert mse < mse_threshold
    assert r2 > r2_threshold
    assert fit_time < fit_time_threshold


@pytest.mark.regression_three_dim
@pytest.mark.parametrize("func", [func_quadratic_3d, func_sin_plus_cos_3d])
def test_regression_3d(func):
    """Test function for regression on 3D functions."""
    run_regression_test_3d(func)
    return
