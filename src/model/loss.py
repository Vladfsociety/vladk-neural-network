class MeanSquaredError:
    def value(self, predicted, actual):
        return 0.5 * ((actual - predicted) ** 2).sum() / predicted.size(0)

    def derivative(self, predicted, actual):
        return predicted - actual