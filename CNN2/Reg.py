import numpy as np
from FCL import FCL

class Reg:
    def __init__(self, m, n):
        self.fcl = FCL(m, n)
        self.last_output = 0
        self.batch_size = 0

    def apply(self, input_vector):
        out = self.fcl.apply(input_vector)
        self.last_output = out
        return out

    def error(self, labels):
        error = np.sum((self.last_output - labels), axis=1)
        return error

    def backprop(self, labels, learn_rate):
        error = (self.last_output - labels)
        return self.fcl.backprop(error, learn_rate)

    def squared_error(self, labels):
        return (0.5) * (np.sum(self.last_output - labels)**2) / 16

