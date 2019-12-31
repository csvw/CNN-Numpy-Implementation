import numpy as np

class FCL:
    '''
    Fully Connected Layer
    '''
    
    def __init__(self, m, n):
        self.weights = np.random.randn(m, n) / m
        self.biases = np.zeros(n)
        self.backprop_data = 0

    def apply(self, input_matrix):
        original_shape = input_matrix.shape
        flattened = input_matrix.flatten()
        results = np.dot(flattened, self.weights) + self.biases 
        self.backprop_data = FCLBackpropData(flattened, original_shape, results)
        return results

    def backprop(self, dLoss_dOutput, learn_rate):
        for i, gradient in enumerate(dLoss_dOutput):
            if gradient == 0:
                continue
            exps = np.exp(self.backprop_data.last_result)
            S = np.sum(exps)

            dOutput_dProbabilities = -exps[i] * exps / (S**2)
            dOutput_dProbabilities[i] = exps[i] * (S - exps[i]) / (S**2)

            dProbabilities_dWeights = self.backprop_data.last_input
            dProbabilities_dBias = 1
            dProbabilities_dInput = self.weights

            dLoss_dProbabilities = gradient * dOutput_dProbabilities

            dLoss_dWeights = dProbabilities_dWeights[np.newaxis].T @ dLoss_dProbabilities[np.newaxis]
            dLoss_dBias = dLoss_dProbabilities * dProbabilities_dBias
            dLoss_dInput = dProbabilities_dInput @ dLoss_dProbabilities

            self.weights -= learn_rate * dLoss_dWeights
            self.biases  -= learn_rate * dLoss_dBias

            return dLoss_dInput.reshape(self.backprop_data.last_input_shape)

class FCLBackpropData:
    def __init__(self, last_input, last_input_shape, last_result):
        self.last_input = last_input
        self.last_input_shape = last_input_shape
        self.last_result = last_result

    