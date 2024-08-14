import pprint
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f
from model.base import NeuralNetwork
from model.layer import FullyConnected
from model.loss import MeanSquaredError
from model.opimizer import SGD
from mpl_toolkits.mplot3d import Axes3D


def func_3d(arg_x, arg_y):
    #return torch.sin(arg_x) + torch.cos(arg_y)
    #return torch.sin(arg_x) * torch.cos(arg_y)
    return 0.2 * arg_x ** 2 + 0.2 * arg_y ** 2

# x = torch.linspace(-5, 5, 100)
# y = torch.linspace(-5, 5, 100)
# x, y = torch.meshgrid(x, y, indexing='ij')
#
# print(x)
# print(y)
#
# # Calculate z values based on the function
# z = func_3d(x, y)
#
# print(z)
#
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the surface
# ax.plot_surface(x, y, z, cmap='viridis')
#
# # Add labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('f(X, Y)')
#
# plt.savefig('3d.jpg')
# # Show the plot
# plt.show()
#
# #sys.exit(0)

def plot(x, y, name):
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    plt.savefig(name)
    plt.show()

def plot_3d(x, y, z, name):

    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    plt.grid(True)
    plt.savefig(name)
    plt.show()


def generate_train_test_data_3d():
    # Create a grid of x and y values
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    x, y = torch.meshgrid(x, y, indexing='ij')

    train_data = []

    for i in range(x.size(0)):
        for j in range(x.size(1)):
            train_data.append({
                'input': [x[i][j], y[i][j]],
                'output': [func_3d(x[i][j], y[i][j])],
            })

    x = torch.linspace(-5, 5, 30)
    y = torch.linspace(-5, 5, 30)
    x, y = torch.meshgrid(x, y, indexing='ij')

    test_data = []

    for i in range(x.size(0)):
        for j in range(x.size(1)):
            test_data.append({
                'input': [x[i][j], y[i][j]],
                'output': [func_3d(x[i][j], y[i][j])],
            })

    return train_data, test_data

def generate_train_test_data():
    x = torch.linspace(-5, 5, 100)

    train_data = []

    def func_quad(arg):
        return 0.5 * arg ** 2 + 2 * arg - 1

    def func_linear(arg):
        return -1.0 * arg - 1

    for x_i in x:
        train_data.append({
            'input': [x_i],
            'output': [func_linear(x_i)],
        })

    x = torch.linspace(-5, 5, 30)

    test_data = []

    for x_i in x:
        test_data.append({
            'input': [x_i],
            'output': [func_linear(x_i)],
        })

    return train_data, test_data


start_time = time.time()

train_dataset, test_dataset = generate_train_test_data()

#train_dataset, test_dataset = generate_train_test_data_3d()

print('train_dataset')
pprint.pprint(train_dataset)
print('test_dataset')
pprint.pprint(test_dataset)

# layers = [
#     FullyConnected(64, f.relu),
#     FullyConnected(64, f.relu),
#     FullyConnected(64, f.relu),
#     FullyConnected(1, 'linear')
# ]
layers = [
    FullyConnected(128, f.relu),
    FullyConnected(128, f.relu),
    FullyConnected(128, f.relu),
    FullyConnected(128, f.relu),
    FullyConnected(1, 'linear')
]
# layers = [
#     FullyConnected(256, f.relu),
#     FullyConnected(256, f.relu),
#     FullyConnected(256, f.relu),
#     FullyConnected(256, f.relu),
#     FullyConnected(256, f.relu),
#     FullyConnected(256, f.relu),
#     FullyConnected(1, 'linear')
# ]
# layers = [
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(512, f.relu),
#     FullyConnected(1, 'linear')
# ]

nn = NeuralNetwork(1, layers, optimizer=SGD(), loss=MeanSquaredError())
#nn = NeuralNetwork(2, layers, optimizer=SGD(), loss=MeanSquaredError())

epochs = 100
nn.fit(train_dataset, test_dataset, epochs=epochs)

prediction = nn.predict(test_dataset)

print("--- %s seconds ---" % (time.time() - start_time))

plot([d['input'][0].item() for d in test_dataset], [pred[0].item() for pred in prediction], 'test_result.jpg')
plot([d['input'][0].item() for d in train_dataset], [d['output'][0].item() for d in train_dataset], 'train.jpg')

# plot_3d(
#     torch.tensor([[train_dataset[j*100 + i]['input'][0].item() for i in range(100)] for j in range(100)]),
#     torch.tensor([[train_dataset[j*100 + i]['input'][1].item() for i in range(100)] for j in range(100)]),
#     torch.tensor([[train_dataset[j*100 + i]['output'][0].item() for i in range(100)] for j in range(100)]),
#     'train_3d.jpg'
# )
# plot_3d(
#     torch.tensor([[test_dataset[j*30 + i]['input'][0].item() for i in range(30)] for j in range(30)]),
#     torch.tensor([[test_dataset[j*30 + i]['input'][1].item() for i in range(30)] for j in range(30)]),
#     torch.tensor([[prediction[j*30 + i][0].item() for i in range(30)] for j in range(30)]),
#     'test_result_3d.jpg'
# )
