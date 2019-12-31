import numpy as np
from Convolution import Convolution
from Maxpool import Maxpool
from FCL import FCL
from Regression import Regression
from Relu import Relu
from pathlib import Path

# Batch:  (labels, tensor)
# tensor: ()

class NeuralNetworkMR:
    def __init__(self):
        self.convolution = Convolution(3, 8)
        self.pool = Maxpool()
        self.fcl = FCL(11 * 23 * 8, 128)
        self.fcl1 = FCL(128, 128)
        #self.fcl2 = FCL(64, 64)
        self.relu = Relu()
        self.relu1 = Relu()
        #self.relu2 = Relu()
        self.regression = Regression(11 * 23 * 8, 5)

    def normalize(self, image):
        return (image /255) - 0.5

    def normalize_batch(self, batch):
        batch[1] = (batch[1]/255) - 0.5
        return batch

    def forward_train(self, im, label):
        im  = self.normalize(im)
        out = self.convolution.apply(im)
        out = self.pool.apply(out)
        out = self.fcl.apply(out)
        out = self.relu.apply(out)
        out = self.fcl1.apply(out)
        out = self.relu.apply(out)
        out = self.regression.apply(out)
        #print(out.shape)

        #print("labels shape: " + str(label.shape))
        
        loss = np.sum(self.regression.squared_error(label))
        acc  = np.abs(self.regression.error(label))

        return out, loss, acc

    def forward(self, im):
        im  = self.normalize(im)
        out = self.convolution.apply(im)
        out = self.pool.apply(out)
        out = self.fcl.apply(out)
        out = self.relu.apply(out)
        out = self.regression.apply(out)

        return out

    def forward_batch(self, batch):
        batch = self.normalize_batch(batch)
        out = self.convolution.apply_batch(batch[1])
        out = self.pool.apply_batch(out)
        #out = self.fcl.apply_batch(out)
        #out = self.relu.apply_batch(out)
        #out = self.fcl1.apply_batch(out)
        #out = self.relu1.apply_batch(out)
        #out = self.fcl2.apply_batch(out)
        #out = self.relu2.apply_batch(out)
        out = self.regression.apply_batch(out)

        labels = batch[0]

        #print("labels:")
        #print(labels.shape)

        loss = np.sum(self.regression.squared_error_backprop(labels))
        acc  = np.abs(self.regression.error_backprop(labels))

        return out, loss, acc

    def train_batch(self, batch, lr=0.005):
        out, loss, acc = self.forward_batch(batch)
                
        labels = batch[0]
        gradient = self.regression.backprop_batch(labels, lr)
        #gradient = self.relu2.backprop_batch(gradient)
        #gradient = self.fcl2.backprop_batch(gradient, lr)
        #gradient = self.relu1.backprop_batch(gradient)
        #gradient = self.fcl1.backprop_batch(gradient, lr)
        #gradient = self.relu.backprop(gradient)
        #gradient = self.fcl.backprop_batch(gradient, lr)
        #gradient = self.pool.backprop_batch(gradient)
        #self.convolution.backprop_batch(gradient, lr)

        return out, loss, acc

    # def grad_check(self, batch):
    #     batch = self.normalize_batch(batch)
    #     out = self.convolution.apply_batch(batch[1])
    #     out = self.pool.apply_batch(out)
    #     out0 = self.regression.apply_batch(out)
    #     error = self.regression.error(batch[0])
    #     gradient = error.T @ self.regression.fcl.last_input / self.regression.fcl.last_input_shape[0]
    #     N, M =  self.regression.fcl.weights.shape
    #     for i in range(N):
    #         for j in range(M):
    #             self.regression.fcl.weights[i][j] += 0.0001
    #             batch = self.normalize_batch(batch)
    #             out = self.convolution.apply_batch(batch[1])
    #             out = self.pool.apply_batch(out)
    #             out1 = self.regression.apply_batch(out)
    #             self.regression.fcl.weights[i][j] -= 0.0002
    #             out = self.convolution.apply_batch(batch[1])
    #             out = self.pool.apply_batch(out)
    #             out2 = self.regression.apply_batch(out)
                
    #             self.regression.fcl.weights[i][j] += 0.0001
    #             res = (out1 - out2) / (0.0002)
    #             print()
    #             print(res.shape)
    #             print(out1.shape)
    #             print(gradient.shape)
    #             for k in range(out1.shape[0]):
    #                 print(str(res[k][0]) + "|||" + str(gradient[k][0]))



    def train(self, label, im, lr=0.00005):
        out, loss, acc = self.forward_train(im, label)

        gradient = self.regression.backprop(label, lr)
        gradient = self.relu.backprop(gradient)
        gradient = self.fcl.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        self.convolution.backprop(gradient, lr)

        return out, loss, acc