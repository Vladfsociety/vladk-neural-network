import time
import matplotlib.pyplot as plt
import torch
from src.model.base import NeuralNetwork
from src.model.layer import Input, FullyConnected
from src.model.loss import MeanSquaredError
from src.model.optimizer import SGD
from src.model.activation import Relu, Linear
from src.model.metric import R2Score


def func_3d(arg_x, arg_y):
    # return torch.sin(arg_x) + torch.cos(arg_y)
    # return torch.sin(arg_x) * torch.cos(arg_y)
    return 0.2 * arg_x**2 + 0.2 * arg_y**2


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

    # x = torch.tensor(x)
    # y = torch.tensor(y)
    # z = torch.tensor(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x, y, z, cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(X, Y)")

    plt.grid(True)
    plt.savefig(name)
    plt.show()


def generate_train_test_data_3d():
    # Create a grid of x and y values
    x = torch.linspace(-5, 5, 50)
    y = torch.linspace(-5, 5, 50)
    x, y = torch.meshgrid(x, y, indexing="ij")

    train_data = []

    for i in range(x.size(0)):
        for j in range(x.size(1)):
            train_data.append(
                {
                    "input": [x[i][j], y[i][j]],
                    "output": [func_3d(x[i][j], y[i][j])],
                }
            )

    x = torch.linspace(-5, 5, 30)
    y = torch.linspace(-5, 5, 30)
    x, y = torch.meshgrid(x, y, indexing="ij")

    test_data = []

    for i in range(x.size(0)):
        for j in range(x.size(1)):
            test_data.append(
                {
                    "input": [x[i][j], y[i][j]],
                    "output": [func_3d(x[i][j], y[i][j])],
                }
            )

    return train_data, test_data


def generate_train_test_data():
    x = torch.linspace(-5, 5, 100)

    train_data = []

    def func_quad(arg):
        return 0.5 * arg**2 + 2 * arg - 1

    def func_linear(arg):
        return -1.0 * arg - 1

    for x_i in x:
        train_data.append(
            {
                "input": [x_i],
                "output": [func_quad(x_i)],
            }
        )

    x = torch.linspace(-5, 5, 30)

    test_data = []

    for x_i in x:
        test_data.append(
            {
                "input": [x_i],
                "output": [func_quad(x_i)],
            }
        )

    return train_data, test_data


start_time = time.time()

train_dataset, test_dataset = generate_train_test_data()

# train_dataset, test_dataset = generate_train_test_data_3d()

# print('train_dataset')
# pprint.pprint(train_dataset)
# print('test_dataset')
# pprint.pprint(test_dataset)

layers = [
    FullyConnected(128, Relu()),
    FullyConnected(128, Relu()),
    FullyConnected(128, Relu()),
    FullyConnected(1, Linear()),
]
# layers = [
#     FullyConnected(128, Relu()),
#     FullyConnected(128, Relu()),
#     FullyConnected(128, Relu()),
#     FullyConnected(128, Relu()),
#     FullyConnected(1, Linear())
# ]
# layers = [
#     FullyConnected(256, Relu()),
#     FullyConnected(256, Relu()),
#     FullyConnected(256, Relu()),
#     FullyConnected(256, Relu()),
#     FullyConnected(256, Relu()),
#     FullyConnected(256, Relu()),
#     FullyConnected(1, Linear())
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

nn = NeuralNetwork(
    Input(1), layers, optimizer=SGD(), loss=MeanSquaredError(), metric=R2Score()
)
# nn = NeuralNetwork(
#     Input(2),
#     layers,
#     optimizer=SGD(learning_rate=0.001),
#     loss=MeanSquaredError(),
#     metric=R2Score()
# )

epochs = 40
nn.fit(train_dataset, test_dataset, epochs=epochs, batch_size=1, verbose=True)

prediction = nn.predict(test_dataset)

print("--- %s seconds ---" % (time.time() - start_time))

plot(
    [d["input"][0].item() for d in test_dataset],
    [pred[0].item() for pred in prediction],
    "test_result.jpg",
)
plot(
    [d["input"][0].item() for d in train_dataset],
    [d["output"][0].item() for d in train_dataset],
    "train.jpg",
)

# plot_3d(
#     torch.tensor([[train_dataset[j*50 + i]['input'][0].item() for i in range(50)] for j in range(50)]),
#     torch.tensor([[train_dataset[j*50 + i]['input'][1].item() for i in range(50)] for j in range(50)]),
#     torch.tensor([[train_dataset[j*50 + i]['output'][0].item() for i in range(50)] for j in range(50)]),
#     'train_3d.jpg'
# )
# plot_3d(
#     torch.tensor([[test_dataset[j*30 + i]['input'][0].item() for i in range(30)] for j in range(30)]),
#     torch.tensor([[test_dataset[j*30 + i]['input'][1].item() for i in range(30)] for j in range(30)]),
#     torch.tensor([[prediction[j*30 + i][0].item() for i in range(30)] for j in range(30)]),
#     'test_result_3d.jpg'
# )
