import torch


class SGD:

    def __init__(self, learning_rate=0.001):
        self.__learning_rate = learning_rate

    def initialize(self, layers):
        return

    def update(self, layers, delta_w, delta_b, batch_size):

        layer_index = len(layers) - 1

        while layer_index > 0:

            layers[layer_index]["w"] -= (self.__learning_rate / batch_size) * delta_w[
                layer_index - 1
            ]
            layers[layer_index]["b"] -= (self.__learning_rate / batch_size) * delta_b[
                layer_index - 1
            ]

            layer_index -= 1

        return


class Adam:

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

        for layer in layers[1:]:

            self.__first_moment_w.append(torch.zeros_like(layer["w"]))
            self.__first_moment_b.append(torch.zeros_like(layer["b"]))
            self.__second_moment_w.append(torch.zeros_like(layer["w"]))
            self.__second_moment_b.append(torch.zeros_like(layer["b"]))

        return

    def update(self, layers, delta_w, delta_b, batch_size):

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

        first_moment_w = self.__first_moment_w[layer_index - 1]
        first_moment_w = (
            self.__beta_1 * first_moment_w
            + (1.0 - self.__beta_1) * delta_w[layer_index - 1]
        )

        self.__first_moment_w[layer_index - 1] = first_moment_w

        return first_moment_w / (1.0 - self.__beta_1**self.__timestamp)

    def __get_second_moment_w(self, layer_index, delta_w):

        second_moment_w = self.__second_moment_w[layer_index - 1]
        second_moment_w = self.__beta_2 * second_moment_w + (1.0 - self.__beta_2) * (
            delta_w[layer_index - 1] ** 2
        )

        self.__second_moment_w[layer_index - 1] = second_moment_w

        return second_moment_w / (1.0 - self.__beta_2**self.__timestamp)

    def __get_first_moment_b(self, layer_index, delta_b):

        first_moment_b = self.__first_moment_b[layer_index - 1]
        first_moment_b = (
            self.__beta_1 * first_moment_b
            + (1.0 - self.__beta_1) * delta_b[layer_index - 1]
        )

        self.__first_moment_b[layer_index - 1] = first_moment_b

        return first_moment_b / (1.0 - self.__beta_1**self.__timestamp)

    def __get_second_moment_b(self, layer_index, delta_b):

        second_moment_b = self.__second_moment_b[layer_index - 1]
        second_moment_b = self.__beta_2 * second_moment_b + (1.0 - self.__beta_2) * (
            delta_b[layer_index - 1] ** 2
        )

        self.__second_moment_b[layer_index - 1] = second_moment_b

        return second_moment_b / (1.0 - self.__beta_2**self.__timestamp)
