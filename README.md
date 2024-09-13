# Custom Neural Network Library

This repository is a mini-library that implements a simple version of a feedforward neural network (FNN) and convolutional neural network (CNN) from scratch using Python and PyTorch. PyTorch is used solely for mathematical and element-wise operations on tensors (without using autograd), and for speeding up computations by utilizing the GPU. Simply put, PyTorch is used as a replacement for NumPy but with GPU acceleration. The library provides basic functionalities for building, training, and evaluating custom neural network models for both regression and classification tasks.

## Features

- **Feedforward Neural Network (FNN)**: A fully connected neural network suitable for regression and classification tasks.
- **Convolutional Neural Network (CNN)**: A simplified CNN implementation with convolutional and max-pooling layers.
- **Activation Functions**: Includes ReLU, Leaky ReLU, Sigmoid, and Linear activations.
- **Loss Functions**: Support for mean squared error (MSE), binary cross-entropy (BCE) and categorical cross-entropy (CCE) losses.
- **Optimizers**: Implementations of stochastic gradient descent (SGD) and Adam optimizers.
- **Metrics**: Accuracy metric for classification tasks, especially useful for one-hot encoded data, and R2 score for regression tasks.

## Installation
```
pip install vladk-neural-network
```

## Usage

### Data Format examples:
#### Example for regression:
```
# sample shape (2, 1) - 2 input values, 1 output value
dataset = [
    {
        "input": [0.1, 0.2],
        "output": [0.15],
    },
    {
        "input": [0.8, 0.9],
        "output": [0.7],
    },
]
```
#### Example for classification, output values one-hot encoded:
```
# sample shape (4, 2) - 4 input values, 2 output one-hot encoded values
dataset = [
    {
        "input": [0.13, 0.22, 0.37, 0.41],
        "output": [1.0, 0.0],
    },
    {
        "input": [0.76, 0.87, 0.91, 0.93],
        "output": [0.0, 1.0],
    },
]
```

### Model creation examples:
#### Feedforward Neural Network for regression:

```
from vladk_neural_network.model.activation import Linear, Relu
from vladk_neural_network.model.base import NeuralNetwork
from vladk_neural_network.model.layer import FullyConnected, Input
from vladk_neural_network.model.loss import MeanSquaredError
from vladk_neural_network.model.metric import R2Score
from vladk_neural_network.model.optimizer import SGD

# Build model
layers = [
    FullyConnected(64, Relu()),
    FullyConnected(64, Relu()),
    FullyConnected(1, Linear()),
]
nn = NeuralNetwork(
    Input(2),
    layers,
    optimizer=SGD(),
    loss=MeanSquaredError(),
    metric=R2Score()
)

# Train model
history = nn.fit(train_dataset, test_dataset, epochs=20, batch_size=1, verbose=True)

# Using model for prediction
prediction = nn.predict(test_dataset)
```
#### Convolutional Neural Network for classification:
```
from vladk_neural_network.model.activation import LeakyRelu, Linear
from vladk_neural_network.model.base import NeuralNetwork
from vladk_neural_network.model.layer import (
    Convolutional,
    Flatten,
    FullyConnected,
    Input3D,
    MaxPool2D,
)
from vladk_neural_network.model.loss import CategoricalCrossEntropy
from vladk_neural_network.model.metric import AccuracyOneHot
from vladk_neural_network.model.optimizer import Adam

# Build model using gpu acceleration and applying argmax convert to raw prediction probabilities
layers = [
    Convolutional(LeakyRelu(), filters_num=4, kernel_size=3, padding_type="same"),
    Convolutional(LeakyRelu(), filters_num=8, kernel_size=3),
    Convolutional(LeakyRelu(), filters_num=16, kernel_size=3),
    MaxPool2D(),
    Flatten(),
    FullyConnected(64, LeakyRelu()),
    FullyConnected(10, Linear()),
]
cnn = NeuralNetwork(
    Input3D((1, 28, 28)),
    layers,
    optimizer=Adam(),
    loss=CategoricalCrossEntropy(),
    metric=AccuracyOneHot(),
    convert_prediction='argmax',
    use_gpu=True
)

# Train model
cnn.fit(train_dataset, test_dataset, epochs=10, batch_size=1, verbose=True)

# Using model for prediction
prediction = cnn.predict(test_dataset)
```
Several examples, including training feedforward and convolutional neural networks, are available in the form of Jupyter notebooks in the **notebooks/** folder. You can view and run these examples to understand how to use the library for different tasks.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.