import numpy as np

# Citation: gradient for Relu obtained from the following article:
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

class Relu:
    def __init__(self):
        self.last_input = 0
        self.last_batch = 0

    def apply(self, input_vector):
        self.last_input = input_vector
        input_vector[input_vector <= 0] = 0
        return input_vector

    def apply_batch(self, input_tensor):
        self.last_batch = input_tensor
        #print("Changed? " + str(np.sum(input_tensor[0])))
        for sample in input_tensor:
            sample[sample <= 0] = 0
        #print("Changed: " + str(np.sum(input_tensor[0])))
        return input_tensor

    def backprop(self, dLoss_dOutput):
        dLoss_dOutput[self.last_input <= 0] = 0
        return dLoss_dOutput

    def backprop_batch(self, dLoss_dOutput):
        for img in range(self.last_batch.shape[0]):
            sample = self.last_batch[img]
            dLoss_dOutput[img][sample <= 0] = 0
        return dLoss_dOutput