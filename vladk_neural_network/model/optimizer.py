import torch


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(self, learning_rate=0.001):
        self._learning_rate = learning_rate

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

            layers[layer_index + 1].w -= (
                self._learning_rate / batch_size
            ) * layer.grad_w
            layers[layer_index + 1].b -= (
                self._learning_rate / batch_size
            ) * layer.grad_b

        return


class Adam:
    """
    Adam optimizer.

    Performs parameter updates using the Adam optimization algorithm with momentum and adaptive learning rates.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, weight_decay=0.0):
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._timestamp = 0
        self._first_moment_w = []
        self._first_moment_b = []
        self._second_moment_w = []
        self._second_moment_b = []
        self._correction_1 = None
        self._correction_2 = None

    def initialize(self, layers):
        """
        Initialize the optimizer for the given layers by setting up moment estimates.
        This includes initializing first and second moment estimates for weights and biases.
        """
        for layer in layers[1:]:
            if layer.learnable:
                # Initialize first and second moment estimates for weights and biases
                self._first_moment_w.append(torch.zeros_like(layer.w))
                self._first_moment_b.append(torch.zeros_like(layer.b))
                self._second_moment_w.append(torch.zeros_like(layer.w))
                self._second_moment_b.append(torch.zeros_like(layer.b))
            else:
                # For non-learnable layers, add placeholder zero tensors
                self._first_moment_w.append(torch.zeros(1))
                self._first_moment_b.append(torch.zeros(1))
                self._second_moment_w.append(torch.zeros(1))
                self._second_moment_b.append(torch.zeros(1))

        return

    def update(self, layers, batch_size):
        """
        Update parameters using Adam.
        """
        self._timestamp += 1

        self._correction_1 = 1.0 - self._beta_1**self._timestamp
        self._correction_2 = 1.0 - self._beta_2**self._timestamp

        for layer_index, layer in enumerate(layers[1:]):
            if not layer.learnable:
                continue

            grad_w_average = layer.grad_w / batch_size

            if self._weight_decay != 0.0:
                grad_w_average += self._weight_decay * layers[layer_index + 1].w

            first_moment_w = self._get_first_moment_w(layer_index, grad_w_average)
            second_moment_w = self._get_second_moment_w(layer_index, grad_w_average)

            layers[layer_index + 1].w -= (self._learning_rate * first_moment_w) / (
                torch.sqrt(second_moment_w) + self._epsilon
            )

            grad_b_average = layer.grad_b / batch_size
            first_moment_b = self._get_first_moment_b(layer_index, grad_b_average)
            second_moment_b = self._get_second_moment_b(layer_index, grad_b_average)

            layers[layer_index + 1].b -= (self._learning_rate * first_moment_b) / (
                torch.sqrt(second_moment_b) + self._epsilon
            )

        return

    def _get_first_moment_w(self, layer_index, grad_w):
        first_moment_w = self._first_moment_w[layer_index]
        first_moment_w = self._beta_1 * first_moment_w + (1.0 - self._beta_1) * grad_w
        self._first_moment_w[layer_index] = first_moment_w

        return first_moment_w / self._correction_1

    def _get_second_moment_w(self, layer_index, grad_w):
        second_moment_w = self._second_moment_w[layer_index]
        second_moment_w = self._beta_2 * second_moment_w + (1.0 - self._beta_2) * (
            grad_w**2
        )
        self._second_moment_w[layer_index] = second_moment_w

        return second_moment_w / self._correction_2

    def _get_first_moment_b(self, layer_index, grad_b):
        first_moment_b = self._first_moment_b[layer_index]
        first_moment_b = self._beta_1 * first_moment_b + (1.0 - self._beta_1) * grad_b
        self._first_moment_b[layer_index] = first_moment_b

        return first_moment_b / self._correction_1

    def _get_second_moment_b(self, layer_index, grad_b):
        second_moment_b = self._second_moment_b[layer_index]
        second_moment_b = self._beta_2 * second_moment_b + (1.0 - self._beta_2) * (
            grad_b**2
        )
        self._second_moment_b[layer_index] = second_moment_b

        return second_moment_b / self._correction_2
