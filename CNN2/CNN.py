import numpy as np
from pathlib import Path
from Conv import Conv
from FCL import FCL
from Pool import Pool
from Reg import Reg
from Relu import Relu
from Softmax import Softmax

class CNNMR:
    def __init__(self):
        self.conv1 = Conv(3, 8, 1)
        self.relu1 = Relu()
        self.pool1 = Pool()
        self.conv2 = Conv(3, 16, 8)
        self.relu2 = Relu()
        self.pool2 = Pool()
        #self.fcl1  = FCL(12 * 6 * 16, 120)
        #self.fcl2  = FCL(120, 80)
        #self.relu3 = Relu()
        #self.relu4 = Relu()
        #self.relu = Relu()
        #self.fcl1  = FCL(256, 256)
        #self.relu1 = Relu()
        self.fcl  = Reg(6 * 6 * 16, 2)
        self.soft = Softmax()

    def normalize_batch(self, batch):
        batch[1] = (batch[1]/255) - 0.5
        if batch[1].ndim != 4:
            batch[1] = batch[1][:,:, np.newaxis, :]
        #print(batch[1].shape)
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
        out = self.conv2.apply(out)
        out = self.relu2.apply(out)
        out = self.pool2.apply(out)
        # out = self.fcl1.apply(out)
        # out = self.relu3.apply(out)
        # out = self.fcl2.apply(out)
        # out = self.relu4.apply(out)
        # out = self.reg.apply(out)
        out = self.fcl.apply(out)
        out = self.soft.apply(out)
        return out

    def train(self, batch, lr=0.05):
        out, loss, acc = self.forward(batch)

        # gradient = self.reg.backprop(batch[0], lr)
        # gradient = self.relu4.backprop(gradient)
        # gradient = self.fcl2.backprop(gradient, lr)
        # gradient = self.relu3.backprop(gradient)
        # gradient = self.fcl1.backprop(gradient, lr)
        gradient = self.soft.backprop(batch[0])
        gradient = self.fcl.backprop(gradient, lr)
        gradient = self.pool2.backprop(gradient)
        gradient = self.relu2.backprop(gradient)
        gradient = self.conv2.backprop(gradient, lr)
        gradient = self.pool1.backprop(gradient)
        gradient = self.relu1.backprop(gradient)
        gradient = self.conv1.backprop(gradient, lr)
        #gradient = self.conv.backprop(gradient, lr)
        

        return out, loss, acc

    # def test(self, batch):
    #     self.conv.test_conv_forward(batch[1])
    #     out = self.conv.apply(batch[1])
    #     self.pool.test_pool_forward(out)
    #     out = self.pool.apply(out)
    #     #self.fcl.test_fcl_forward(out)
    #     out = self.fcl.apply(out)
    #     out = self.relu.apply(out)
    #     out = self.reg.apply(out)