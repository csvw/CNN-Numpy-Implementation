import numpy as np
from Convolution import Convolution
from Maxpool import Maxpool
from Softmax import Softmax
from FCLSotmax import FCL
import csv
from pathlib import Path

class NeuralNetwork:
    def __init__(self):
        self.conv_layer = Convolution(3, 8)
        self.pool_layer = Maxpool()
        self.fcl_layer  = FCL(13 * 13 * 8, 10)
        self.soft_layer = Softmax()
        self.dLoss_dOut_cross_entropy = 0

    def normalize(self, image):
        return (image /255) - 0.5

    def forward(self, image, label):
        im  = self.normalize(image)
        out = self.conv_layer.apply(im)
        out = self.pool_layer.apply(out)
        out = self.fcl_layer.apply(out)
        out = self.soft_layer.apply(out)
        
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        self.dLoss_dOut_cross_entropy = np.zeros(10)
        self.dLoss_dOut_cross_entropy[label] = -1. / out[label]

        return out, loss, acc

    def train(self, im, label, lr=0.005):
        out, loss, acc = self.forward(im, label)

        gradient = self.dLoss_dOut_cross_entropy
        gradient = self.fcl_layer.backprop(gradient, lr)
        gradient = self.pool_layer.backprop(gradient)
        gradient = self.conv_layer.backprop(gradient, lr)

        return loss, acc

    
            