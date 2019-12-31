import numpy as np

class Soft:
    def __init__(self):
        self.X = 0
        self.P = 0

    def forward(self, X):
        self.X = X
        
        # X_stable = X - np.max(X)
        # print("X", X)
        exps = np.exp(X)
        S = np.sum(exps, axis=1, keepdims=True)
        P = exps / (S)
        # print("P", P)
        # print(S)
        self.P = P
        # print(P)

        # print(P)

        return P


    def predict(self):
        pass

    # Citations:
    # https://deepnotes.io/softmax-crossentropy
    # https://victorzhou.com/blog/intro-to-cnns-part-2/
    def cross_entropy_loss(self, label):
        '''
        One hot encoded vector means that only one probability will be selected.
        '''

        P = self.P
        Y = np.array(label)
        l = np.argmax(Y)

        # print(P.shape)
        # print(P[0][l])

        L = -np.log(P[0][l])

        # print(L)

        return L

    def backprop(self, Y):
        '''
        (Input_Size X Batch_Size)
        '''
        #print(self.P.shape)
        #print(Y.shape)
        # print(Y)
        P = self.P
        G = P - Y
        # print(G)
        # print(G)
        # print("Gradient", G)
        #print(G)
        # P = self.P
        # G = np.zeros(P.shape)
        # for y in range(len(Y)):
        #     if Y[y] == 1:
        #         G[0][y] = (-1./P[0][y])

        return G

    def acc(self, Y):
        # print(self.last_output.shape)
        # print(label.shape)

        # print(label)
        # print(self.last_output)
        # print(np.argmax(self.last_output, axis=1) == np.argmax(label, axis=1))

        same = float(np.sum(np.argmax(self.P) == np.argmax(Y)).astype(int))
        
        return same
        
    def kronecker(self, i, j):
        return int(i == j)
