import numpy as np
from FCL import FCL

class Regression:
    def __init__(self, m, n):
        self.fcl = FCL(m, n)
        self.last_output = 0
        self.batch_size = 0

    def apply(self, input_vector):
        out = self.fcl.apply(input_vector)
        self.last_output = out
        return out

    def apply_batch(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        out = self.fcl.apply_batch(input_tensor)
        self.last_output = out
        return out

    def error(self, labels):
        error = (self.last_output - labels)
        return error

    def error_backprop(self, labels):
        # print(self.last_output.shape)
        # print(labels.shape)
        error = np.sum((self.last_output - labels), axis=0)
        #print(error)
        # error_batch = np.zeros((self.batch_size, labels.shape[1]))
        # print("Error")
        # print(error.shape)
        # print(error_batch.shape)
        # for b in range(self.batch_size):
        #     error_batch[b] = error
        # print("Error shape " + str(error_batch.shape))
        return error

    def backprop(self, labels, learn_rate):
        error = (self.last_output - labels)
        #print(error)
        return self.fcl.backprop(error, learn_rate) 

    def backprop_batch(self, labels, learn_rate):
        error = (self.last_output - labels)
        #print(error)
        # for b in range(len(error)):
        #     print(self.last_output[b])
        #     print(labels[b])
        #     print(error[b])
        # error_batch = np.zeros((self.batch_size, labels.shape[1]))
        # for b in range(self.batch_size):
        #     error_batch[b] = error
        # print("Error shape " + str(error_batch.shape))
        return self.fcl.backprop_batch(error, learn_rate)

    def squared_error_backprop(self, labels):
        return (0.5) * (np.sum(self.last_output - labels)**2) / self.batch_size

    def squared_error(self, labels):
        return (0.5) * ((self.last_output - labels)**2)
