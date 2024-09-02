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

    def update(self, layers, delta_w, delta_b, batch_size):
        """
        Update parameters using SGD.
        """
        layer_index = len(layers) - 1

        # print('delta_w____________________________________')
        # print(delta_w)
        # print('delta_b')
        # print(delta_b)

        while layer_index > 0:

            # print('layer_index_______________________________________________')
            # print(layer_index)

            if not layers[layer_index]['learnable']:
                layer_index -= 1
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

            layers[layer_index]["w"] -= (self.__learning_rate / batch_size) * delta_w[
                layer_index - 1
            ]
            layers[layer_index]["b"] -= (self.__learning_rate / batch_size) * delta_b[
                layer_index - 1
            ]

            layer_index -= 1

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
        for layer in layers[1:]:
            self.__first_moment_w.append(torch.zeros_like(layer["w"]))
            self.__first_moment_b.append(torch.zeros_like(layer["b"]))
            self.__second_moment_w.append(torch.zeros_like(layer["w"]))
            self.__second_moment_b.append(torch.zeros_like(layer["b"]))

        return

    def update(self, layers, delta_w, delta_b, batch_size):
        """
        Update parameters using Adam.
        """
        self.__timestamp += 1

        layer_index = len(layers) - 1

        while layer_index > 0:
            first_moment_w = self.__get_first_moment_w(layer_index, delta_w)
            second_moment_w = self.__get_second_moment_w(layer_index, delta_w)

            layers[layer_index]["w"] -= (self.__learning_rate * first_moment_w) / (
                batch_size * (torch.sqrt(second_moment_w) + self.__epsilon)
            )

            first_moment_b = self.__get_first_moment_b(layer_index, delta_b)
            second_moment_b = self.__get_second_moment_b(layer_index, delta_b)

            layers[layer_index]["b"] -= (self.__learning_rate * first_moment_b) / (
                batch_size * (torch.sqrt(second_moment_b) + self.__epsilon)
            )

            layer_index -= 1

        return

    def __get_first_moment_w(self, layer_index, delta_w):
        """
        Compute the first moment estimate for weights.
        """
        first_moment_w = self.__first_moment_w[layer_index - 1]
        first_moment_w = (
            self.__beta_1 * first_moment_w
            + (1.0 - self.__beta_1) * delta_w[layer_index - 1]
        )

        self.__first_moment_w[layer_index - 1] = first_moment_w

        return first_moment_w / (1.0 - self.__beta_1**self.__timestamp)

    def __get_second_moment_w(self, layer_index, delta_w):
        """
        Compute the second moment estimate for weights.
        """
        second_moment_w = self.__second_moment_w[layer_index - 1]
        second_moment_w = self.__beta_2 * second_moment_w + (1.0 - self.__beta_2) * (
            delta_w[layer_index - 1] ** 2
        )

        self.__second_moment_w[layer_index - 1] = second_moment_w

        return second_moment_w / (1.0 - self.__beta_2**self.__timestamp)

    def __get_first_moment_b(self, layer_index, delta_b):
        """
        Compute the first moment estimate for biases.
        """
        first_moment_b = self.__first_moment_b[layer_index - 1]
        first_moment_b = (
            self.__beta_1 * first_moment_b
            + (1.0 - self.__beta_1) * delta_b[layer_index - 1]
        )

        self.__first_moment_b[layer_index - 1] = first_moment_b

        return first_moment_b / (1.0 - self.__beta_1**self.__timestamp)

    def __get_second_moment_b(self, layer_index, delta_b):
        """
        Compute the second moment estimate for biases.
        """
        second_moment_b = self.__second_moment_b[layer_index - 1]
        second_moment_b = self.__beta_2 * second_moment_b + (1.0 - self.__beta_2) * (
            delta_b[layer_index - 1] ** 2
        )

        self.__second_moment_b[layer_index - 1] = second_moment_b

        return second_moment_b / (1.0 - self.__beta_2**self.__timestamp)
