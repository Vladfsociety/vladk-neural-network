import random
import time

import pandas as pd
import pytest
import torch

from vladk_neural_network.model.activation import LeakyRelu, Linear
from vladk_neural_network.model.base import NeuralNetwork
from vladk_neural_network.model.layer import FullyConnected, Input
from vladk_neural_network.model.loss import CategoricalCrossEntropy
from vladk_neural_network.model.metric import AccuracyOneHot
from vladk_neural_network.model.optimizer import Adam

"""
Multi-class classification on the Iris dataset.
"""


def get_iris_dataset():
    """Load and preprocess the Iris dataset for multi-class classification."""
    data = pd.read_csv("data/iris/Iris.csv")
    data.drop("Id", axis=1, inplace=True)
    feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    for feature_column in feature_columns:
        data[feature_column] = (data[feature_column] - data[feature_column].min()) / (
            data[feature_column].max() - data[feature_column].min()
        )

    one_hot_map = {
        "Iris-setosa": [1.0, 0.0, 0.0],
        "Iris-versicolor": [0.0, 1.0, 0.0],
        "Iris-virginica": [0.0, 0.0, 1.0],
    }

    dataset = []
    for index in data.index:
        input_values = [float(val) for val in data.loc[index].drop("Species").values]
        specie = one_hot_map[data.loc[index, "Species"]]
        dataset.append(
            {"input": torch.tensor(input_values), "output": torch.tensor(specie)}
        )

    random.seed(3)
    random.shuffle(dataset)
    return dataset[:115], dataset[115:]


def run_iris_test(
    learning_rate=0.001, cce_threshold=0.1, acc_threshold=0.95, fit_time_threshold=4.0
):
    """Run a multi-class classification test on the Iris dataset."""
    print("\nMulti-class classification. Testing on full Iris dataset (3 species)")

    train_dataset, test_dataset = get_iris_dataset()

    layers = [
        FullyConnected(128, LeakyRelu()),
        FullyConnected(128, LeakyRelu()),
        FullyConnected(128, LeakyRelu()),
        FullyConnected(3, Linear()),
    ]
    nn = NeuralNetwork(
        Input(4),
        layers,
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossEntropy(),
        metric=AccuracyOneHot(),
        convert_prediction="argmax",
    )

    start_time = time.time()
    epochs = 30
    nn.fit(train_dataset, epochs=epochs, batch_size=1, verbose=False)
    fit_time = round(time.time() - start_time, 4)

    prediction, raw_prediction = nn.predict(test_dataset, with_raw_prediction=True)
    actual = torch.stack(
        [
            torch.tensor(test_sample["output"]).unsqueeze(1)
            for test_sample in test_dataset
        ]
    )

    cce = nn.loss(raw_prediction, actual)
    acc = nn.metric(prediction, actual)

    print(f"CCE: {cce}, Accuracy: {acc}, Fit time: {fit_time} seconds")

    assert cce < cce_threshold
    assert acc > acc_threshold
    assert fit_time < fit_time_threshold


@pytest.mark.multi_classification_iris
def test_multi_classification_iris():
    """Test function for multi-class classification on the Iris dataset."""
    run_iris_test()


"""
Multi-class classification on the Digits dataset.
"""


def get_digits_dataset():
    """Load and preprocess the Digits dataset for multi-class classification."""
    dataset = []
    train = pd.read_csv("data/digits/train.csv", header=0, nrows=5000)

    output = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def rotate_vector(vector, rotate_by):
        """Rotate a one-hot encoded vector to match the digit label."""
        return vector[-rotate_by:] + vector[:-rotate_by]

    for index in train.index:
        input_values = [
            float(val) / 255.0 for val in train.loc[index].drop("label").values
        ]
        dataset.append(
            {
                "input": torch.tensor(input_values),
                "output": torch.tensor(
                    rotate_vector(output, int(train.loc[index]["label"]))
                ),
            }
        )

    random.seed(1)
    random.shuffle(dataset)
    return dataset[:2000], dataset[2000:2500]


def run_digits_test(
    learning_rate=0.001, cce_threshold=0.6, acc_threshold=0.92, fit_time_threshold=20.0
):
    """Run a multi-class classification test on the Digits dataset."""
    print("\nMulti-class classification. Testing on Digits dataset")

    train_dataset, test_dataset = get_digits_dataset()

    layers = [
        FullyConnected(256, LeakyRelu()),
        FullyConnected(128, LeakyRelu()),
        FullyConnected(64, LeakyRelu()),
        FullyConnected(10, Linear()),
    ]
    nn = NeuralNetwork(
        Input(784),
        layers,
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossEntropy(),
        metric=AccuracyOneHot(),
        convert_prediction="argmax",
    )

    start_time = time.time()
    epochs = 10
    nn.fit(train_dataset, epochs=epochs, batch_size=8, verbose=False)
    fit_time = round(time.time() - start_time, 4)

    prediction, raw_prediction = nn.predict(test_dataset, with_raw_prediction=True)
    actual = torch.stack(
        [
            torch.tensor(test_sample["output"]).unsqueeze(1)
            for test_sample in test_dataset
        ]
    )

    cce = nn.loss(raw_prediction, actual)
    acc = nn.metric(prediction, actual)

    print(f"CCE: {cce}, Accuracy: {acc}, Fit time: {fit_time} seconds")

    assert cce < cce_threshold
    assert acc > acc_threshold
    assert fit_time < fit_time_threshold


@pytest.mark.multi_classification_digits
def test_multi_classification_digits():
    """Test function for multi-class classification on the Digits dataset."""
    run_digits_test()
