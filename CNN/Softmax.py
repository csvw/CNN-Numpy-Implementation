import numpy as np

class Softmax:
    def apply(self, input_vector):
        exp_vector = np.exp(input_vector)
        probabilities = exp_vector / np.sum(exp_vector, axis=0)
        return probabilities
        
