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
    #plt.show()


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

plot_data(data, labels, 'classification_random_all.jpg')
plot_data(train_data, train_labels, 'classification_random_train.jpg')
plot_data(test_data, test_labels, 'classification_random_test.jpg')

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

# train_dataset = [
#     {
#         'input': [0.1],
#         'output': [0.0]
#     },
#     {
#         'input': [0.9],
#         'output': [1.0]
#     },
#     {
#         'input': [0.75],
#         'output': [1.0]
#     },
#     {
#         'input': [0.3],
#         'output': [0.0]
#     },
#     {
#         'input': [0.7],
#         'output': [1.0]
#     },
#     {
#         'input': [0.12],
#         'output': [0.0]
#     },
#     {
#         'input': [0.43],
#         'output': [0.0]
#     },
#     {
#         'input': [0.57],
#         'output': [1.0]
#     },
#     {
#         'input': [0.35],
#         'output': [0.0]
#     },
#     {
#         'input': [0.67],
#         'output': [1.0]
#     },
#     {
#         'input': [0.25],
#         'output': [0.0]
#     },
#     {
#         'input': [0.78],
#         'output': [1.0]
#     },
#     {
#         'input': [0.29],
#         'output': [0.0]
#     },
#     {
#         'input': [0.63],
#         'output': [1.0]
#     },
# ]
#
# test_dataset = [
#     {
#         'input': [0.2],
#         'output': [0.0]
#     },
#     {
#         'input': [0.85],
#         'output': [1.0]
#     },
#     {
#         'input': [0.32],
#         'output': [0.0]
#     },
#     {
#         'input': [0.71],
#         'output': [1.0]
#     },
# ]

# pred = torch.tensor([[0.4839],
#         [0.4822],
#         [0.4829],
#         [0.4841],
#         [0.4838],
#         [0.4822]])
# act = torch.tensor([[1.],
#         [0.],
#         [0.],
#         [1.],
#         [1.],
#         [0.]])
#
# eps = 0.001
# res = -(act * torch.log(pred + eps) + (torch.ones_like(pred) - act) *
#                    torch.log(torch.ones_like(pred) - pred + eps))
# print(pred - torch.ones_like(pred) + eps)
# print(torch.log(pred - torch.ones_like(pred) + eps))
#
# sys.exit(0)

start_time = time.time()

layers = [
    FullyConnected(128, LeakyRelu()),
    FullyConnected(128, LeakyRelu()),
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
    metric=Accuracy()
)

epochs = 100
nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=4, verbose=True)

prediction = nn.predict(test_dataset)

#plot_data(test_data, prediction, 'classification_random_prediction.jpg')

print("--- %s seconds ---" % (time.time() - start_time))
