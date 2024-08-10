class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self._gradients = []


    def update(self, layers, loss):
        pass