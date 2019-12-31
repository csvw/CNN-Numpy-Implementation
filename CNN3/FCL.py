import numpy as np
import math

class FCL:
    def __init__(self, m, n):
        self.W       = np.random.normal(0, 1, (m, n)) / math.sqrt(m)
        self.b       = np.zeros((1, n))
        self.X       = 0
        self.X_shape = 0
        self.Z       = 0
        self.Vdw = 0
        self.Sdw = 0
        self.Vdb = 0
        self.Sdb = 0
        self.t   = 1

    def flatten(self, X):
        self.X_shape = X.shape
        self.X = X.flatten()
        self.X = self.X[np.newaxis,:]
        return self.X

    def forward(self, X):
        W = self.W
        b = self.b
        X = self.flatten(X)

        # print(X.shape)
        # print(W.shape)

        Z = np.dot(X, W) + b

        self.Z = Z

        return Z

    def adam(self, dW, db, lr):
        b1 = 0.9
        b2 = 0.999
        epsilon = 0.000000001
        self.Vdw = b1 * self.Vdw + (1-b1) * dW
        self.Vdb = b1 * self.Vdb + (1-b1) * db
        self.Sdw = b2 * self.Sdw + (1-b2) * (dW**2)
        self.Sdb = b2 * self.Sdb + (1-b2) * (db**2)
        self.Vdw = self.Vdw / (1 - b1**self.t)
        self.Vdb = self.Vdb / (1 - b1**self.t)
        self.Sdw = self.Sdw / (1 - b2**self.t)
        self.Sdb = self.Sdb / (1 - b2**self.t)

        self.t += 1

        dW = lr * (self.Vdw / (np.sqrt(self.Sdw) + epsilon))
        db = lr * (self.Vdb / (np.sqrt(self.Sdb) + epsilon))

        return dW, db
        

    def backprop(self, dZ, lr):
        W        = self.W
        X        = self.X

        # print(X.shape)
        # print(dZ.shape)

        dA       = np.dot(W, dZ.T) # Not sure
        dW       = np.outer(X, dZ)
        db       = np.sum(dZ, axis=0, keepdims=True)
        
        self.b  -= lr * db
        self.W  -= lr * dW

        # dW, db = self.adam(dW, db, lr)
        # self.b -= db
        # self.W -= dW


        return dA.reshape(self.X_shape)