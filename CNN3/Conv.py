import numpy as np

class Conv:
    def __init__(self, filter_size, num_filters, num_channels):
        self.S = filter_size
        self.N = num_filters
        self.C = num_channels
        self.F = np.random.randn(self.N, self.S, self.S, self.C) * 0.01
        self.b = np.zeros((1, self.N))
        self.X = 0

    def pad(self, X):
        '''
        Will break if you change the filter size.
        '''
        if X.ndim == 2:
            return np.pad(X, ((1, 1), (1, 1)), 'constant')
        else:
            return np.pad(X, ((1, 1), (1, 1), (0, 0)), 'constant')



    def iterate(self, X):
        h, w, c = X.shape
        for i in range(h):
            for j in range(w):
                    yield i, j

    def forward(self, X):
        '''
        Get the dimensions before you pad. Pass the old X. [ X ( A A A ) X ]
        Pixel (i, j) of channel c = Sum of the elementwise product of F and R
        '''
        self.X  = X
        h, w, c = X.shape
        s       = self.S
        Z       = np.zeros((h, w, self.N))
        X       = self.pad(X)

        for i, j in self.iterate(self.X):
            region   = X[i:(i+s), j:(j+s)]
            Z[i, j]  = np.sum(region * self.F, axis=(1, 2, 3))

        return Z

    def backprop(self, dZ, lr):
        '''
        The gradient of the output with respect to the weights, dZdW,
        is the pixel values of the image times the gradient dZ for
        that pixel location.
        For: Z = Sum(Sum(im(i, j)*F(i,j)))
        The gradient of Z with respect to the input is just F, so
        the loss with respect to the input is dZ * F.
        The double sum over every filter location * the gradient dZ for that location.
        For every filter.
        dA should have the same number of channels as the original input.
        '''
        dF = np.zeros(self.F.shape)
        dA = np.zeros(self.X.shape)
        db = np.zeros(self.b.shape)
        X  = self.pad(self.X)
        s  = self.S

        for i, j in self.iterate(self.X):
            region = X[i:(i+s), j:(j+s)]
            for f in range(self.N):
            # print(dF.shape)
            # print(dZ.shape)
            # print(region.shape)
                dF[f] += region * dZ[i, j, f]
                db[0, f]  += dZ[i, j, f]
                if i + s < dA.shape[0] and j + s < dA.shape[1]:
                    dA[i:i+s, j:j+s] += self.F[f] * dZ[i, j, f]

        self.F -= lr * dF
        self.b -= lr * db

        # print(self.F[0][0])

        return dA


