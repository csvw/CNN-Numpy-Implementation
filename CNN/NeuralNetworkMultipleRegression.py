import numpy as np
from Convolution import Convolution
from Maxpool import Maxpool
from Softmax import Softmax
from FCL import FCL
from Regression import Regression
from Relu import Relu
from pathlib import Path

class NeuralNetworkMR:
    def __init__(self):
        self.convolution = Convolution(3, 6)
        self.pool = Maxpool()
        self.fcl = FCL(127 * 159 * 6, 64)
        self.relu = Relu()
        self.regression = Regression(64, 30)

    def normalize(self, image):
        return (image /255) - 0.5

    def forward(self, image, label):
        im  = self.normalize(image)
        #print(im.shape)
        out = self.convolution.apply(im)
        #print(out.shape)
        out = self.pool.apply(out)
        #print(out.shape)
        out = self.fcl.apply(out)
        #print(out.shape)
        out = self.relu.apply(out)
        #print(out.shape)
        out = self.regression.apply(out)
        #print(out.shape)

        #print("labels shape: " + str(label.shape))
        
        loss = -np.sum(self.regression.squared_error(label))
        acc  = np.sum(np.abs(self.regression.error(label)))

        return out, loss, acc

    def train(self, im, label, lr=0.005):
        out, loss, acc = self.forward(im, label)

        gradient = self.regression.backprop(label, lr)
        gradient = self.relu.backprop(gradient)
        gradient = self.fcl.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        self.convolution.backprop(gradient, lr)

        return out, loss, acc