import numpy as np

# Citation: gradient for Relu obtained from the following article:
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
# https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

class Relu:
    def __init__(self):
        self.X = 0

    def forward(self, X):
        self.X = X
        X[X<0] = 0
        return X

    # Every gradient is another possible point of failure...
    def backprop(self, dZ):
        X = self.X
        dZ[X<0] = 0
        return dZ