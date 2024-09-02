import torch


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Performs parameter updates using the SGD optimization algorithm.
    """

    def __init__(self, learning_rate=0.001):
        self.__learning_rate = learning_rate

    def initialize(self, layers):
        """
        Initialize the optimizer for the given layers. For SGD, no initialization is needed.
        """
        return

    def update(self, layers, batch_size):
        """
        Update parameters using SGD.
        """
        for layer_index, layer in enumerate(layers[1:]):

            if not layer.learnable:
                continue

            # print('layers[layer_index]["w"]')
            # print(layers[layer_index]["w"].shape)
            # print(layers[layer_index]["w"])
            # print('(self.__learning_rate / batch_size) * delta_w[layer_index - 1]')
            # print(((self.__learning_rate / batch_size) * delta_w[
            #     layer_index - 1
            # ]).shape)
            # print((self.__learning_rate / batch_size) * delta_w[
            #     layer_index - 1
            # ])
            # print('layers[layer_index]["b"]')
            # print(layers[layer_index]["b"].shape)
            # print(layers[layer_index]["b"])
            # print('(self.__learning_rate / batch_size) * delta_b[layer_index - 1]')
            # print(((self.__learning_rate / batch_size) * delta_b[
            #     layer_index - 1
            # ]).shape)
            # print((self.__learning_rate / batch_size) * delta_b[
            #     layer_index - 1
            # ])

            layers[layer_index + 1].w -= ((self.__learning_rate / batch_size)
                                         * layer.grad_w)
            layers[layer_index + 1].b -= ((self.__learning_rate / batch_size)
                                         * layer.grad_b)

        return


class Adam:
    """
    Adam optimizer.

    Performs parameter updates using the Adam optimization algorithm with momentum and adaptive learning rates.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06):
        self.__learning_rate = learning_rate
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__epsilon = epsilon
        self.__timestamp = 0
        self.__first_moment_w = []
        self.__first_moment_b = []
        self.__second_moment_w = []
        self.__second_moment_b = []

    def initialize(self, layers):
        """
        Initialize the optimizer for the given layers by setting up moment estimates.
        """
        # print('layers')
        # print(layers)

        for layer in layers[1:]:

            # print('layer')
            # print(layer)

            if layer.learnable:
                self.__first_moment_w.append(torch.zeros_like(layer.w))
                self.__first_moment_b.append(torch.zeros_like(layer.b))
                self.__second_moment_w.append(torch.zeros_like(layer.w))
                self.__second_moment_b.append(torch.zeros_like(layer.b))
            else:
                self.__first_moment_w.append(torch.zeros(1))
                self.__first_moment_b.append(torch.zeros(1))
                self.__second_moment_w.append(torch.zeros(1))
                self.__second_moment_b.append(torch.zeros(1))

        return

    def update(self, layers, batch_size):
        """
        Update parameters using Adam.
        """
        self.__timestamp += 1

        for layer_index, layer in enumerate(layers[1:]):

            if not layer.learnable:
                continue

            first_moment_w = self.__get_first_moment_w(layer_index, layer.grad_w)
            second_moment_w = self.__get_second_moment_w(layer_index, layer.grad_w)

            layers[layer_index + 1].w -= (self.__learning_rate * first_moment_w) / (
                batch_size * (torch.sqrt(second_moment_w) + self.__epsilon)
            )

            first_moment_b = self.__get_first_moment_b(layer_index, layer.grad_b)
            second_moment_b = self.__get_second_moment_b(layer_index, layer.grad_b)

            layers[layer_index + 1].b -= (self.__learning_rate * first_moment_b) / (
                batch_size * (torch.sqrt(second_moment_b) + self.__epsilon)
            )

        return

    def __get_first_moment_w(self, layer_index, grad_w):
        """
        Compute the first moment estimate for weights.
        """
        first_moment_w = self.__first_moment_w[layer_index]
        first_moment_w = (
            self.__beta_1 * first_moment_w
            + (1.0 - self.__beta_1) * grad_w
        )

        self.__first_moment_w[layer_index] = first_moment_w

        return first_moment_w / (1.0 - self.__beta_1 ** self.__timestamp)

    def __get_second_moment_w(self, layer_index, grad_w):
        """
        Compute the second moment estimate for weights.
        """
        second_moment_w = self.__second_moment_w[layer_index]
        second_moment_w = self.__beta_2 * second_moment_w + (1.0 - self.__beta_2) * (
            grad_w ** 2
        )

        self.__second_moment_w[layer_index] = second_moment_w

        return second_moment_w / (1.0 - self.__beta_2 ** self.__timestamp)

    def __get_first_moment_b(self, layer_index, grad_b):
        """
        Compute the first moment estimate for biases.
        """
        first_moment_b = self.__first_moment_b[layer_index]
        first_moment_b = (
            self.__beta_1 * first_moment_b
            + (1.0 - self.__beta_1) * grad_b
        )

        self.__first_moment_b[layer_index] = first_moment_b

        return first_moment_b / (1.0 - self.__beta_1 ** self.__timestamp)

    def __get_second_moment_b(self, layer_index, grad_b):
        """
        Compute the second moment estimate for biases.
        """
        second_moment_b = self.__second_moment_b[layer_index]
        second_moment_b = self.__beta_2 * second_moment_b + (1.0 - self.__beta_2) * (
            grad_b ** 2
        )

        self.__second_moment_b[layer_index] = second_moment_b

        return second_moment_b / (1.0 - self.__beta_2 ** self.__timestamp)
