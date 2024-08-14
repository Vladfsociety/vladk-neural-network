class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self._gradients = []

    def update_w(self, layer, previous_layer, error):
        return layer['w'] - self.learning_rate * (previous_layer['a'] @ error)

    def update_b(self, layer, error):
        return layer['b'] - self.learning_rate * error
