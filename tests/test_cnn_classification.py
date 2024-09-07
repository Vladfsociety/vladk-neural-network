import random
import time

import pandas as pd
import pytest
import torch

from src.model.activation import LeakyRelu, Linear
from src.model.base import NeuralNetwork
from src.model.layer import FullyConnected, Input3D, Convolutional, Flatten
from src.model.loss import CategoricalCrossEntropy
from src.model.metric import AccuracyOneHot
from src.model.optimizer import Adam

"""
Multi-class classification on the Digits dataset (CNN model).
"""

def get_onehot_digit(digit):
    output = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return output[-digit:] + output[:-digit]


def get_digits_dataset():
    """Load and preprocess the Digits dataset for multi-class classification."""
    dataset = []

    train = pd.read_csv("data/digits/train.csv", header=0, nrows=5000)

    for index in train.index:
        input_values = [
            float(val) / 255.0 for val in train.loc[index].drop("label").values
        ]
        dataset.append(
            {
                "input": [torch.tensor(input_values).reshape(28, 28).tolist()],
                "output": get_onehot_digit(int(train.loc[index]["label"])),
            }
        )

    random.seed(5)
    random.shuffle(dataset)
    return dataset[:2000], dataset[2000:2500]


def run_digits_test(
    learning_rate=0.001, cce_threshold=1.65, acc_threshold=0.96, fit_time_threshold=80.0
):
    """Run a multi-class classification test on the Digits dataset."""
    print("\nMulti-class classification. Testing on Digits dataset (CNN)")

    train_dataset, test_dataset = get_digits_dataset()

    layers = [
        Convolutional(LeakyRelu(), filters_num=4, kernel_size=3),
        Convolutional(LeakyRelu(), filters_num=8, kernel_size=3),
        Convolutional(LeakyRelu(), filters_num=16, kernel_size=3),
        Flatten(),
        FullyConnected(128, LeakyRelu()),
        FullyConnected(10, Linear())
    ]
    cnn = NeuralNetwork(
        Input3D((1, 28, 28)),
        layers,
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossEntropy(),
        metric=AccuracyOneHot(),
        convert_prediction='argmax',
        use_gpu=True
    )

    if str(cnn.device) == "cpu":
        fit_time_threshold = 180.0

    start_time = time.time()
    epochs = 10
    cnn.fit(train_dataset, epochs=epochs, batch_size=1, verbose=False)
    fit_time = round(time.time() - start_time, 4)

    prediction, raw_prediction = cnn.predict(test_dataset, with_raw_prediction=True)
    actual = torch.stack(
        [
            torch.tensor(test_sample["output"]).unsqueeze(1)
            for test_sample in test_dataset
        ]
    )

    cce = cnn.loss(raw_prediction, actual)
    acc = cnn.metric(prediction, actual)

    print(f"CCE: {cce}, Accuracy: {acc}, Fit time: {fit_time} seconds")

    assert cce < cce_threshold
    assert acc > acc_threshold
    assert fit_time < fit_time_threshold


@pytest.mark.cnn_multi_classification_digits
def test_multi_classification_digits():
    """Test function for multi-class classification on the Digits dataset."""
    run_digits_test()
