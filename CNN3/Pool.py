import numpy as np

class Pool:
    def __init__(self):
        self.X_argmax = 0
        self.X        = 0

    def iterate(self, X):
        h, w, c = X.shape
        for i in range(h // 2):
            for j in range(w // 2):
                yield i, j

    def forward(self, X):
        self.X   = X
        h, w, c  = X.shape
        Z        = np.zeros((h // 2, w // 2, c))
        X_argmax = np.zeros((h // 2, w // 2, c))

        for i, j in self.iterate(X):
            region   = X[i*2:(i+1)*2, j*2:(j+1)*2]
            Z[i, j]  = np.max(region, axis=(0, 1))

        return Z

    def backprop(self, dZ):
        '''
        Iterate over half the image's original dimensions.
        Check each 2x2 region in that image.
        Find the indices with the max pixel for that region.
        Set dA's pixel value to the value for dZ.
        Leave the others black.
        '''
        h, w, c = self.X.shape
        dA = np.zeros(self.X.shape)

        for i, j in self.iterate(self.X):
            region    = self.X[i*2:(i+1)*2, j*2:(j+1)*2]
            max_per_F = np.max(region, axis=(0,1))
            for f in range(c):
                for i1 in range(2):
                    for j1 in range(2):
                        if max_per_F[f] == region[i1, j1, f]:
                            dA[i*2+i1, j*2+j1, f] = dZ[i, j, f]
        
        return dA
