import pytest
import time
import random
import torch
import pandas as pd

from src.model.base import NeuralNetwork
from src.model.layer import Input, FullyConnected
from src.model.loss import BinaryCrossEntropy
from src.model.optimizer import SGD
from src.model.metric import Accuracy
from src.model.activation import LeakyRelu, Sigmoid


"""
Binary classification on iris dataset
"""
def get_iris_dataset(species_to_compare, specie_to_exclude):
    data = pd.read_csv('data/iris/Iris.csv')
    data.drop('Id', axis=1, inplace=True)
    data = data[data['Species'] != specie_to_exclude]
    with pd.option_context('future.no_silent_downcasting', True):
        data['Species'] = data['Species'].replace(species_to_compare[0], 0)
        data['Species'] = data['Species'].replace(species_to_compare[1], 1)
    data['Species'] = data['Species'].astype('float32')
    feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    for feature_column in feature_columns:
        data[feature_column] = ((data[feature_column] - data[feature_column].min())
                                / (data[feature_column].max() - data[feature_column].min()))
        data[feature_column] = data[feature_column].astype('float32')

    dataset = []
    for index in data.index:
        dataset.append({
            'input': data.loc[index].drop('Species').values,
            'output': [float(data.loc[index, 'Species'])]
        })

    random.shuffle(dataset)
    return dataset[:70], dataset[70:]

def run_classification_test(
    species_combination,
    learning_rate=0.002,
    bce_threshold=0.1,
    acc_threshold=0.98,
    fit_time_threshold=2.0
):
    print(f"\nBinary classification. Testing {species_combination[0][0]} - {species_combination[0][1]} combination")

    train_dataset, test_dataset = get_iris_dataset(species_combination[0], species_combination[1])

    layers = [
        FullyConnected(128, LeakyRelu()),
        FullyConnected(128, LeakyRelu()),
        FullyConnected(128, LeakyRelu()),
        FullyConnected(1, Sigmoid())
    ]
    nn = NeuralNetwork(
        Input(4),
        layers,
        optimizer=SGD(learning_rate=learning_rate),
        loss=BinaryCrossEntropy(),
        metric=Accuracy(),
        convert_prediction='binary'
    )

    start_time = time.time()
    epochs = 50
    nn.fit(train_dataset,  epochs=epochs, batch_size=1, verbose=False)
    fit_time = round(time.time() - start_time, 4)

    prediction, raw_prediction = nn.predict(test_dataset, with_raw_prediction=True)
    actual = torch.stack([torch.tensor(test_sample['output']).unsqueeze(1) for test_sample in test_dataset])

    bce = nn.loss(raw_prediction, actual)
    acc = nn.metric(prediction, actual)

    print(f"BCE: {bce}, Accuracy: {acc}, Fit time: {fit_time}")

    assert bce < bce_threshold
    assert acc > acc_threshold
    assert fit_time < fit_time_threshold

@pytest.mark.binary_classification_iris
@pytest.mark.parametrize("species_combination, bce_threshold, acc_threshold", [
    ([['Iris-setosa', 'Iris-versicolor'], 'Iris-virginica'], 0.1, 0.98),
    ([['Iris-versicolor', 'Iris-virginica'], 'Iris-setosa'], 0.5, 0.85),
    ([['Iris-setosa', 'Iris-virginica'], 'Iris-versicolor'], 0.1, 0.98)
])
def test_classification(species_combination, bce_threshold, acc_threshold):
    run_classification_test(species_combination, bce_threshold=bce_threshold, acc_threshold=acc_threshold)
    return