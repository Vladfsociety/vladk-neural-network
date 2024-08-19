class SGD:

    def __init__(self, learning_rate=0.001):
        self.__learning_rate = learning_rate

    def update(self, layers, delta_w, delta_b, batch_size):

        layer_index = len(layers) - 1

        while layer_index > 0:

            layers[layer_index]['w'] -= (self.__learning_rate / batch_size) * delta_w[layer_index - 1]
            layers[layer_index]['b'] -= (self.__learning_rate / batch_size) * delta_b[layer_index - 1]

            layer_index -= 1

        return
