import pprint
import random
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model.base import NeuralNetwork
from src.model.layer import Input, FullyConnected
from src.model.loss import CategoricalCrossEntropy
from src.model.optimizer import SGD, Adam
from src.model.activation import Linear, LeakyRelu, Relu
from src.model.metric import AccuracyOneHot


# def create_classification_dataset(num_samples=1000, noise=0.1, random_seed=42):
#     torch.manual_seed(random_seed)
#
#     data = torch.randn(num_samples, 2)
#
#     labels = torch.zeros(num_samples)
#     for i in range(num_samples):
#         x, y = data[i]
#         distance = math.sqrt(x ** 2 + y ** 2)
#         if distance < 1:
#             labels[i] = 1
#
#     noise_mask = torch.rand(num_samples) < noise
#     labels[noise_mask] = 1 - labels[noise_mask]
#
#     return data / 5, labels
#
# def plot_data(data, labels, name):
#     plt.figure(figsize=(8, 8))
#     colors = []
#     for label in labels:
#         if type(label) == list:
#             label = label[0]
#         colors.append('red' if label == 0 else 'blue')
#     plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
#     plt.title("Simple Classification Dataset")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.savefig(name)
#
#
# # Example usage:
# num_samples = 1300
# noise = 0.1
# random_seed = 42
#
# data, labels = create_classification_dataset(num_samples, noise, random_seed)
#
# train_data = data[:1000]
# train_labels = labels[:1000]
#
# test_data = data[1000:]
# test_labels = labels[1000:]
#
# print(f"Data shape: {data.shape}")
# print(f"Labels shape: {labels.shape}")
#
# plot_data(data, labels, 'classification_synthetic_all.jpg')
# plot_data(train_data, train_labels, 'classification_synthetic_train.jpg')
# plot_data(test_data, test_labels, 'classification_synthetic_test.jpg')
#
# train_dataset = []
# for index in range(len(train_data)):
#     train_dataset.append({
#         'input': train_data[index],
#         'output': train_labels[index]
#     })
#
# test_dataset = []
# for index in range(len(test_data)):
#     test_dataset.append({
#         'input': test_data[index],
#         'output': test_labels[index]
#     })

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
#     return random.shuffle(train_dataset)[:1000], random.shuffle(test_dataset)[:1000] # ?????????????


# train_dataset, test_dataset = generate_train_test_data()

# start_time = time.time()
#
# layers = [
#     FullyConnected(128, LeakyRelu()),
#     FullyConnected(128, LeakyRelu()),
#     FullyConnected(128, LeakyRelu()),
#     FullyConnected(1, Sigmoid())
# ]
# nn = NeuralNetwork(
#     Input(2),
#     layers,
#     optimizer=SGD(learning_rate=0.01),
#     loss=BinaryCrossEntropy(),
#     metric=Accuracy(),
#     convert_prediction='binary'
# )
#
# epochs = 50
# nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=4, verbose=True)
#
# prediction = nn.predict(test_dataset)
#
# plot_data(test_data, prediction, 'classification_synthetic_prediction.jpg')
#
# print("--- %s seconds ---" % (time.time() - start_time))


# def iris_plot(data, species_names, image_name):
#
#     sepal_length, sepal_width, petal_length, petal_width, colors = [], [], [], [], []
#
#     for row in data:
#         sepal_length.append(row['input'][0])
#         sepal_width.append(row['input'][1])
#         petal_length.append(row['input'][2])
#         petal_width.append(row['input'][3])
#         colors.append('green' if row['output'][0] == 0.0 else 'blue')
#
#     plt.figure(figsize=(8, 8))
#     plt.scatter(sepal_length, sepal_width, c=colors, alpha=0.5)
#     green_patch = mpatches.Patch(color='green', label=species_names[0])
#     blue_patch = mpatches.Patch(color='blue', label=species_names[1])
#     plt.legend(handles=[green_patch, blue_patch], title="Species")
#     plt.title(f"Sepal ({species_names[0]}, {species_names[1]})")
#     plt.xlabel("sepal_length")
#     plt.ylabel("sepal_width")
#     plt.savefig('results/iris/' + species_names[0] + '__' + species_names[1] + '/sepal_' + image_name)
#
#     plt.figure(figsize=(8, 8))
#     plt.scatter(petal_length, petal_width, c=colors, alpha=0.5)
#     green_patch = mpatches.Patch(color='green', label=species_names[0])
#     blue_patch = mpatches.Patch(color='blue', label=species_names[1])
#     plt.legend(handles=[green_patch, blue_patch], title="Species")
#     plt.title(f"Petal ({species_names[0]}, {species_names[1]})")
#     plt.xlabel("petal_length")
#     plt.ylabel("petal_width")
#     plt.savefig('results/iris/' + species_names[0] + '__' + species_names[1] + '/petal_' + image_name)
#
#     return
#
# def generate_iris_dataset(species_to_compare, specie_to_exclude):
#     data = pd.read_csv('data/iris/Iris.csv')
#     data.drop('Id', axis=1, inplace=True)
#     data = data[data['Species'] != specie_to_exclude]
#     data['Species'] = data['Species'].replace(species_to_compare[0], 0)
#     data['Species'] = data['Species'].replace(species_to_compare[1], 1)
#     data['Species'] = data['Species'].astype('float32')
#
#     feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
#     for feature_column in feature_columns:
#         data[feature_column] = (data[feature_column] - data[feature_column].min())/(data[feature_column].max() - data[feature_column].min())
#         data[feature_column] = data[feature_column].astype('float32')
#
#     dataset = []
#     for index in data.index:
#         dataset.append({
#             'input': data.loc[index].drop('Species').values,
#             'output': [float(data.loc[index, 'Species'])]
#         })
#
#     random.shuffle(dataset)
#     return dataset[:70], dataset[70:]
#
# #random.seed(43)
#
# species = [
#     [['Iris-setosa', 'Iris-versicolor'], 'Iris-virginica'],
#     [['Iris-versicolor', 'Iris-virginica'], 'Iris-setosa'],
#     [['Iris-setosa', 'Iris-virginica'], 'Iris-versicolor'],
# ]
#
# for specie_comb in species:
#
#     train_dataset, test_dataset = generate_iris_dataset(specie_comb[0], specie_comb[1])
#
#     # print('train_dataset')
#     # pprint.pprint(train_dataset)
#     # print('test_dataset')
#     # pprint.pprint(test_dataset)
#
#     iris_plot(train_dataset + test_dataset, specie_comb[0], 'iris_all.jpg')
#     iris_plot(train_dataset, specie_comb[0], 'iris_train.jpg')
#     iris_plot(test_dataset, specie_comb[0], 'iris_test.jpg')
#
#     start_time = time.time()
#
#     layers = [
#         FullyConnected(128, LeakyRelu()),
#         FullyConnected(128, LeakyRelu()),
#         FullyConnected(128, LeakyRelu()),
#         FullyConnected(1, Sigmoid())
#     ]
#     nn = NeuralNetwork(
#         Input(4),
#         layers,
#         optimizer=SGD(learning_rate=0.002),
#         loss=BinaryCrossEntropy(),
#         metric=Accuracy(),
#         convert_prediction='binary'
#     )
#
#     print(f"{specie_comb[0][0]} - {specie_comb[0][1]}")
#
#     epochs = 40
#     nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=1, verbose=True)
#
#     prediction = nn.predict(test_dataset)
#
#     predicted = test_dataset
#
#     for index, predict in enumerate(prediction):
#         predicted[index]['output'] = predict
#
#     iris_plot(predicted, specie_comb[0], 'iris_predict.jpg')
#
#     print("--- %s seconds ---" % (time.time() - start_time))


# train_dataset = [
#     {
#         'input': [0.1],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.5],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.9],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.2],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.5],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.95],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.1],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.45],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.9],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.3],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.45],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.77],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.05],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.45],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.85],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.15],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.48],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.8],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.3],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.6],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.9],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.1],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.4],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.9],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.1],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.6],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.87],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.3],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.66],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.91],
#         'output': [0.0, 0.0, 1.0]
#     },
#     {
#         'input': [0.31],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.69],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.73],
#         'output': [0.0, 0.0, 1.0]
#     },
# ]
#
# test_dataset = [
#     {
#         'input': [0.1],
#         'output': [1.0, 0.0, 0.0]
#     },
#     {
#         'input': [0.5],
#         'output': [0.0, 1.0, 0.0]
#     },
#     {
#         'input': [0.9],
#         'output': [0.0, 0.0, 1.0]
#     },
# ]

# def iris_plot(data, image_name):
#
#     sepal_length, sepal_width, petal_length, petal_width, colors = [], [], [], [], []
#
#     for row in data:
#         sepal_length.append(row['input'][0])
#         sepal_width.append(row['input'][1])
#         petal_length.append(row['input'][2])
#         petal_width.append(row['input'][3])
#         if row['output'] == [1.0, 0.0, 0.0]:
#             color = 'green'
#         elif row['output'] == [0.0, 1.0, 0.0]:
#             color = 'blue'
#         else:
#             color = 'violet'
#         colors.append(color)
#
#     plt.figure(figsize=(8, 8))
#     plt.scatter(sepal_length, sepal_width, c=colors, alpha=0.5)
#     green_patch = mpatches.Patch(color='green', label='Iris-setosa')
#     blue_patch = mpatches.Patch(color='blue', label='Iris-versicolor')
#     violet_patch = mpatches.Patch(color='violet', label='Iris-virginica')
#     plt.legend(handles=[green_patch, blue_patch, violet_patch], title="Species")
#     plt.title(f"Sepal")
#     plt.xlabel("sepal_length")
#     plt.ylabel("sepal_width")
#     plt.savefig('results/iris/multiclass/sepal_' + image_name)
#
#     plt.figure(figsize=(8, 8))
#     plt.scatter(petal_length, petal_width, c=colors, alpha=0.5)
#     green_patch = mpatches.Patch(color='green', label='Iris-setosa')
#     blue_patch = mpatches.Patch(color='blue', label='Iris-versicolor')
#     violet_patch = mpatches.Patch(color='violet', label='Iris-virginica')
#     plt.legend(handles=[green_patch, blue_patch, violet_patch], title="Species")
#     plt.title(f"Petal")
#     plt.xlabel("petal_length")
#     plt.ylabel("petal_width")
#     plt.savefig('results/iris/multiclass/petal_' + image_name)
#
#     return
#
# def generate_iris_dataset():
#     data = pd.read_csv('data/iris/Iris.csv')
#     data.drop('Id', axis=1, inplace=True)
#     feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
#     for feature_column in feature_columns:
#         data[feature_column] = (data[feature_column] - data[feature_column].min())/(data[feature_column].max() - data[feature_column].min())
#
#     one_hot_map = {
#         'Iris-setosa': [1.0, 0.0, 0.0],
#         'Iris-versicolor': [0.0, 1.0, 0.0],
#         'Iris-virginica': [0.0, 0.0, 1.0],
#     }
#
#     dataset = []
#     for index in data.index:
#         input_values = [float(val) for val in data.loc[index].drop('Species').values]
#         specie = one_hot_map[data.loc[index, 'Species']]
#         dataset.append({
#             'input': input_values,
#             'output': specie
#         })
#
#     random.shuffle(dataset)
#     return dataset[:115], dataset[115:]
#
# #random.seed(45)
#
# train_dataset, test_dataset = generate_iris_dataset()
#
# iris_plot(train_dataset + test_dataset, 'iris_all.jpg')
# iris_plot(train_dataset, 'iris_train.jpg')
# iris_plot(test_dataset, 'iris_test.jpg')
#
# # print('train_dataset')
# # print(train_dataset)
# # print('test_dataset')
# # print(test_dataset)
# #
# # sys.exit(0)
#
# start_time = time.time()
#
# layers = [
#     FullyConnected(128, LeakyRelu()),
#     FullyConnected(128, LeakyRelu()),
#     FullyConnected(128, LeakyRelu()),
#     FullyConnected(3, Linear())
# ]
# nn = NeuralNetwork(
#     Input(4),
#     layers,
#     optimizer=Adam(learning_rate=0.001),
#     #optimizer=SGD(learning_rate=0.001),
#     loss=CategoricalCrossEntropy(),
#     metric=AccuracyOneHot(),
#     convert_prediction='argmax'
# )
#
# epochs = 30
# nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=1, verbose=True)
#
# prediction, raw_pred = nn.predict(test_dataset, with_raw_prediction=True)
#
# predicted = test_dataset.copy()
#
# for index, predict in enumerate(prediction):
#     predicted[index]['output'] = predict.flatten().tolist()
#
# iris_plot(predicted, 'iris_predict.jpg')
#
# print("--- %s seconds ---" % (time.time() - start_time))


def plot_digit(image, index):
    image = torch.tensor(image["input"]).numpy().reshape(28, 28)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title("MNIST Digit:")
    plt.axis("off")  # Hide axis
    plt.show()
    plt.savefig("results/digits/test/" + str(index) + ".jpg")
    # plt.show()

    return


def generate_train_test_data_digits():

    train_dataset = []

    train = pd.read_csv("data/digits/train.csv", header=0, nrows=5000)

    output = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def rotate_vector(vector, rotate_by):
        return vector[-rotate_by:] + vector[:-rotate_by]

    for index in train.index:
        input_values = [
            float(val) / 255.0 for val in train.loc[index].drop("label").values
        ]
        train_dataset.append(
            {
                "input": input_values,
                "output": rotate_vector(output, int(train.loc[index]["label"])),
            }
        )

    random.shuffle(train_dataset)

    return train_dataset[:2000], train_dataset[2000:2500]


random.seed(43)

train_dataset, test_dataset = generate_train_test_data_digits()

start_time = time.time()

layers = [
    FullyConnected(256, LeakyRelu()),
    FullyConnected(128, LeakyRelu()),
    FullyConnected(64, LeakyRelu()),
    FullyConnected(10, Linear()),
]
nn = NeuralNetwork(
    Input(784),
    layers,
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossEntropy(),
    metric=AccuracyOneHot(),
    convert_prediction="argmax",
)

epochs = 20
nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=8, verbose=True)

prediction = nn.predict(test_dataset)

print("--- %s seconds ---" % (time.time() - start_time))
