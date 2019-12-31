import numpy as np
import math

class Conv:
    def __init__(self, dims, num_filters, prev_channels):
        self.filter_dims = dims
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, dims, dims, prev_channels) / (dims ** 2)
        self.bias = np.zeros((num_filters, 1))
        self.filter_edge = int(dims/2)
        self.last_input = 0
        self.last_batch = 0

    def iterate(self, batch):
        for b in range(batch.shape[0]):
            for i in range(batch.shape[1] - 2 * self.filter_edge):
                for j in range(batch.shape[2] - 2 * self.filter_edge):
                    for f in range(self.num_filters):
                        yield b, i, j, f

    def compute_new_dims(self, batch):
        batch_size, dimy, dimx, num_f = batch.shape
        dimy -= 2 * self.filter_edge
        dimx -= 2 * self.filter_edge
        return batch_size, dimy, dimx, num_f

    def apply(self, batch):
        self.last_input = batch
        batch = self.pad_batch(batch)

        batch_size, dimy, dimx, num_f = self.compute_new_dims(batch)
        convolution_cube = np.zeros((batch_size, dimy, dimx, self.num_filters))

        for b, i, j, f in self.iterate(batch):  # batch: FXFXC filter: FXFXC
            convolution_cube[b, i, j, f] = np.sum(batch[b, i:i+self.filter_dims, j:j+self.filter_dims] * self.filters[f]) + self.bias[f]

        return convolution_cube

    def pad_batch(self, batch):
        batch = np.pad(batch, ((0, 0), (1, 1), (1, 1), (0,0)), 'constant')
        return batch

    def backprop(self, dLoss_dOutput, learn_rate):
        dLoss_dFilters = np.zeros(self.filters.shape)
        dLoss_dBias = np.zeros(self.bias.shape)
        dLoss_dInput = np.zeros(self.last_input.shape)

        batch_size, dimy, dimx, num_f = self.last_input.shape

        # print(dLoss_dFilters.shape)
        # print(dLoss_dOutput.shape)

        for i in range(dimy - 2 * self.filter_edge):
            for j in range(dimx - 2 * self.filter_edge):
                for f in range(self.num_filters):
                    region_batch = self.last_input[:, i:i+self.filter_dims, j:j+self.filter_dims, :]
                    for b in range(batch_size): # DLDI: WXHXFXB DLDO: WXHXCXB  Filter by DLDO: (WXHXC) * (1)
                        region = region_batch[b] # for every filter slice (wXhXc) += (wxhxc) * 1
                        dLoss_dFilters[f] += dLoss_dOutput[b, i, j, f] * region / batch_size
                        dLoss_dBias[f] += dLoss_dOutput[b, i, j, f] / batch_size
                        dLoss_dInput[b, i:i+self.filter_dims, j:j+self.filter_dims] += self.filters[f] * dLoss_dOutput[b, i, j, f]

        self.filters -= learn_rate * dLoss_dFilters
        self.bias    -= learn_rate * dLoss_dBias

        #print(self.filters)

        return dLoss_dInput

class Pool:
    def __init__(self):
        self.last_input = 0

    def apply(self, convolution_cube):
        self.last_input = convolution_cube
        batch_size, dimy, dimx, num_filters = convolution_cube.shape
        half_dimy = int(dimy / 2)
        half_dimx = int(dimx / 2)

        out = np.zeros((batch_size, half_dimy, half_dimx, num_filters))

        for i in range(half_dimy):
            for j in range(half_dimx):
                out[:, i, j] = np.amax(convolution_cube[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)], axis = (1, 2))

        return out

    def backprop(self, dLoss_dOutput):
        dLoss_dInput = np.zeros(self.last_input.shape)
        batch_size, dimy, dimx, numf = self.last_input.shape
        half_dimy = dimy // 2
        half_dimx = dimx // 2

        #print(self.last_input.shape)

        for i in range(half_dimy):
            for j in range(half_dimx):
                region_batch = self.last_input[:, (i*2):(i*2+2), (j*2):(j*2+2)]
                for b in range(batch_size):
                    region = region_batch[b]
                    #print(region.shape)
                    max_vec = np.amax(region, axis=(0,1))
                    for i2 in range(2):
                        for j2 in range(2):
                            for f2 in range(numf):
                                if region[i2, j2, f2] == max_vec[f2]:
                                    dLoss_dInput[b, i*2+i2, j*2+j2, f2] = dLoss_dOutput[b, i, j, f2]

        return dLoss_dInput


class Relu:
    def __init__(self):
        self.last_batch = 0

    def apply(self, input_tensor):
        self.last_batch = input_tensor
        input_tensor[input_tensor <= 0] = 0
        return input_tensor

    def backprop(self, dLoss_dOutput):
        dLoss_dOutput[self.last_batch <= 0] = 0
        return dLoss_dOutput

class FCL:
    def __init__(self, m, n):
        self.W = np.random.randn(m, n) * 0.01
        self.b = np.zeros((1, n))
        self.last_input = 0
        self.last_input_shape = 0

    def flatten(self, X):
        #print(X.shape)
        self.last_input_shape = X.shape
        self.last_input = X
        input_flat = X
        if X.ndim > 2:
            input_flat = np.zeros((X.shape[0], self.W.shape[0]))
            for b in range(X.shape[0]):
                input_flat[b] = X[b].flatten()
            self.last_input = input_flat
        return input_flat

    def apply(self, X):
        W = self.W
        b = self.b
        X = self.flatten(X)

        out = np.dot(X, W) + b

        return out

    def backprop(self, dL_dO, lr):
        W   = self.W
        X   = self.last_input

        dL_dI   = np.dot(W, dL_dO.T) # Not sure

        dL_dW   = np.dot(X.T, dL_dO)
        dL_dB   = np.sum(dL_dO, axis=0, keepdims=True)
        self.b  -= lr * dL_dB
        self.W  -= lr * dL_dW

        # print(np.round(self.W[0:10, 0:10], 4))

        return dL_dI.reshape(self.last_input_shape)

class Softmax:
    def __init__(self):
        self.last_input = 0
        self.exps = 0
        self.last_output = 0

    def apply(self, input_vector):
        self.last_input = input_vector
        exp_vector = np.exp(input_vector - np.max(input_vector))
        S = np.sum(exp_vector, axis=1, keepdims=True)
        #print(S)
        probabilities = exp_vector / S
        # print(probabilities)
        # print(input_vector)
        self.last_output = probabilities
        return probabilities

    # Citation: numerically stable sigmoid
    # 
    def sigmoid(self, input_vector):
        result = np.zeros(input_vector.shape)
        for i in range(len(input_vector)):
            x = input_vector[i]
            if x < 0:
                a = math.exp(x) 
                a = a / (1 + a) 
                result[i] = a
            else:
                result[i] = 1 / (1 + math.exp(-x))
        return result
    

    def predict(self):
        pass

    # Citations:
    # https://deepnotes.io/softmax-crossentropy
    # https://victorzhou.com/blog/intro-to-cnns-part-2/
    def cross_entropy_loss(self, label):
        '''
        One hot encoded vector means that only one probability will be selected.
        '''

        probs = np.zeros((self.last_input.shape[0], 1))
        P = self.last_output

        for i in range(self.last_input.shape[0]):
            probs[i] = -np.log(P[i, np.argmax(label[i])])

        return np.sum(probs) / self.last_output.shape[0]

    # def binary_cross_entropy(self, label):
    #     selections = np.argmax(label, axis=1)
    #     P = self.last_output
    #     loss = -np.sum(label * np.log(P) + (1 - label) * np.log(1 - P))
        
    # def sigmoid_backprop(self, label):


    def backprop(self, label):
        '''
        (Input_Size X Batch_Size)

        '''

        P = self.last_output
        Y = label
        G = P - Y
        # print(P)
        # print(Y)
        # print(G)
        # print("P", P)
        # print("G", G)
        return G

    def acc(self, label):
        # print(self.last_output.shape)
        # print(label.shape)

        # print(label)
        # print(self.last_output)
        # print(np.argmax(self.last_output, axis=1) == np.argmax(label, axis=1))

        same = float(np.sum(np.argmax(self.last_output, axis=1) == np.argmax(label, axis=1)).astype(int))
        
        return same / self.last_output.shape[0]
        
    def kronecker(self, i, j):
        return int(i == j)

class CNNMR:
    def __init__(self):
        self.conv1 = Conv(3, 8, 1)
        self.relu1 = Relu()
        self.pool1 = Pool()
        self.fcl1  = FCL(12 * 12 * 8, 240)
        self.relu2 = Relu()
        self.fcl2  = FCL(240, 2)
        self.relu3 = Relu()
        self.fcl3  = FCL(240, 240)
        self.relu4 = Relu()
        self.fcl4  = FCL(240, 2)
        self.soft  = Softmax()

    def normalize_batch(self, batch):
        batch[1] = (batch[1]/255) - 0.5
        if batch[1].ndim != 4:
            batch[1] = batch[1][:,:,:, np.newaxis]
        return batch

    def forward(self, batch):
        batch = self.normalize_batch(batch)
        out   = self.apply(batch[1])

        loss = np.sum(self.soft.cross_entropy_loss(batch[0]))
        acc  = np.abs(self.soft.acc(batch[0]))

        return out, loss, acc

    def apply(self, batch):
        out = self.conv1.apply(batch)
        out = self.relu1.apply(out)
        out = self.pool1.apply(out)
        out = self.fcl1.apply(out)
        out = self.relu2.apply(out)
        out = self.fcl2.apply(out)
        out = self.relu3.apply(out)
        # out = self.fcl3.apply(out)
        # out = self.relu4.apply(out)
        # out = self.fcl4.apply(out)
        out = self.soft.apply(out)
        return out

    def train(self, batch, lr=0.1):
        out, loss, acc = self.forward(batch)
        gradient = self.soft.backprop(batch[0])
        # gradient = self.fcl4.backprop(gradient, lr)
        # gradient = self.relu4.backprop(gradient)
        # gradient = self.fcl3.backprop(gradient, lr)
        gradient = self.relu3.backprop(gradient)
        gradient = self.fcl2.backprop(gradient, lr)
        gradient = self.relu2.backprop(gradient)
        gradient = self.fcl1.backprop(gradient, lr)
        gradient = self.pool1.backprop(gradient)
        gradient = self.relu1.backprop(gradient)
        gradient = self.conv1.backprop(gradient, lr)        

        return out, loss, acc