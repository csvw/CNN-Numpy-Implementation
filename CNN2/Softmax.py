import numpy as np

class Softmax:
    def __init__(self):
        self.last_input = 0
        self.exps = 0
        self.last_output = 0

    def apply(self, input_vector):
        self.last_input = input_vector
        exp_vector = np.exp(input_vector - np.max(input_vector))
        probabilities = exp_vector / np.sum(exp_vector, axis=0)
        self.last_output = probabilities
        return probabilities

    

    # Citations:
    # https://deepnotes.io/softmax-crossentropy
    # https://victorzhou.com/blog/intro-to-cnns-part-2/
    def cross_entropy_loss(self, label):
        '''
        One hot encoded vector means that only one probability will be selected.
        '''

        return -np.sum(np.log(self.last_output) * label) / self.last_output[1]

    def backprop(self, label):
        '''
        (Input_Size X Batch_Size)

        '''

        P = self.last_output
        Y = label
        G = P - Y
        print("P", P)
        print("G", G)
        return G

    def acc(self, label):
        # print(self.last_output.shape)
        # print(label.shape)

        same = float(np.sum(np.argmax(self.last_output, axis=0) == np.argmax(label, axis=0)).astype(int))
        
        return same / self.last_output.shape[1]
        
    def kronecker(self, i, j):
        return int(i == j)
