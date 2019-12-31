import numpy as np
from PIL import Image
from FCL import FCL
from Convolution import Convolution
from Maxpool import Maxpool

class NeuralNetworkMIMR:
    def __init__(self):
        self.convolution = Convolution(3, 8)
        self.pool = Maxpool()
        self.fcl = FCL(11 * 11 * 8, 128)
        self.relu = Relu()
        self.regression = Regression(128, )