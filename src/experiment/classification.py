import pprint
import sys
import random
import time
import math
import matplotlib.pyplot as plt
import torch

from src.model.base import NeuralNetwork
from src.model.layer import Input, FullyConnected
from src.model.loss import BinaryCrossEntropy
from src.model.optimizer import SGD
from src.model.activation import Relu, Sigmoid, LeakyRelu
from src.model.metric import Accuracy


def create_classification_dataset(num_samples=1000, noise=0.1, random_seed=42):
    torch.manual_seed(random_seed)

    data = torch.randn(num_samples, 2)

    labels = torch.zeros(num_samples)
    for i in range(num_samples):
        x, y = data[i]
        distance = math.sqrt(x ** 2 + y ** 2)
        if distance < 1:
            labels[i] = 1

    noise_mask = torch.rand(num_samples) < noise
    labels[noise_mask] = 1 - labels[noise_mask]

    return data / 5, labels

def plot_data(data, labels, name):
    plt.figure(figsize=(8, 8))
    colors = []
    for label in labels:
        if type(label) == list:
            label = label[0]
        colors.append('red' if label == 0 else 'blue')
    plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
    plt.title("Simple Classification Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(name)


# Example usage:
num_samples = 1300
noise = 0.1
random_seed = 42

data, labels = create_classification_dataset(num_samples, noise, random_seed)

train_data = data[:1000]
train_labels = labels[:1000]

test_data = data[1000:]
test_labels = labels[1000:]

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

plot_data(data, labels, 'classification_synthetic_all.jpg')
plot_data(train_data, train_labels, 'classification_synthetic_train.jpg')
plot_data(test_data, test_labels, 'classification_synthetic_test.jpg')

train_dataset = []
for index in range(len(train_data)):
    train_dataset.append({
        'input': train_data[index],
        'output': train_labels[index]
    })

test_dataset = []
for index in range(len(test_data)):
    test_dataset.append({
        'input': test_data[index],
        'output': test_labels[index]
    })

# def generate_train_test_data_digits():
#
#     train_dataset = []
#     test_dataset = []
#
#     train = pd.read_csv('data/digits/train.csv', header=0)
#     test = pd.read_csv('data/digits/test.csv', header=0)
#
#     for index in train.index:
#         train_dataset.append({
#             'input': train.loc[index].drop('label').values,
#             'output': train.loc[index]['label']
#         })
#
#     print('train_dataset')
#     print(train_dataset)
#
#     for index in test.index:
#         test_dataset.append({
#             'input': test.loc[index].values
#         })
#
#     print('test_dataset')
#     print(test_dataset)
#
#     return random.shuffle(train_dataset)[:1000], random.shuffle(test_dataset)[:1000]


#train_dataset, test_dataset = generate_train_test_data()

# print('train_dataset')
# print(train_dataset)
# print('test_dataset')
# print(test_dataset)

start_time = time.time()

layers = [
    FullyConnected(128, LeakyRelu()),
    FullyConnected(128, LeakyRelu()),
    FullyConnected(128, LeakyRelu()),
    FullyConnected(1, Sigmoid())
]
nn = NeuralNetwork(
    Input(2),
    layers,
    optimizer=SGD(learning_rate=0.01),
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    convert_prediction='binary'
)

epochs = 50
nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=4, verbose=True)

prediction = nn.predict(test_dataset)

plot_data(test_data, prediction, 'classification_synthetic_prediction.jpg')

print("--- %s seconds ---" % (time.time() - start_time))
