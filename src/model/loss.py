class MeanSquaredError:
    def __init__(self):
        pass

    def calculate(self, predicted, actual):
        return ((actual - predicted)**2).sum()/predicted.size(0)
