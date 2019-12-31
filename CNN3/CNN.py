import numpy as np
from Conv import Conv
from Pool import Pool
from Relu import Relu
from FCL import FCL
from Soft import Soft

class CNNMR:
    def __init__(self):
        self.conv1 = Conv(3, 8, 1)
        # self.relu1 = Relu()
        self.pool1 = Pool()
        # self.conv2 = Conv(3, 16, 8)
        # self.relu2 = Relu()
        # self.pool2 = Pool()
        # self.conv3 = Conv(3, 24, 16)
        # self.pool3 = Pool()
        # self.conv4 = Conv(3, 30, 24)
        # self.pool4 = Pool()
        self.fcl1  = FCL(24 * 24 * 8, 10)
        self.soft  = Soft()

    def normalize_batch(self, batch):
        batch[1] = (batch[1]/255.) - 0.5
        if batch[1].ndim != 3:
            batch[1] = batch[1][:, :, np.newaxis]
        # print(batch[1].shape)
        return batch

    def forward(self, batch):
        # batch = self.normalize_batch(batch)
        #print(batch[1].shape)
        out  = self.apply(batch[1])

        loss = np.sum(self.soft.cross_entropy_loss(batch[0]))
        acc  = np.abs(self.soft.acc(batch[0]))

        return out, loss, acc

    def apply(self, batch):
        out = self.conv1.forward(batch)
        # print(np.max(out))
        # out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        # out = self.conv2.forward(out)
        # out = self.relu2.forward(out)
        # out = self.pool2.forward(out)
        # out = self.conv3.forward(out)
        # out = self.pool3.forward(out)
        # out = self.conv4.forward(out)
        # out = self.pool4.forward(out)
        out = self.fcl1.forward(out)
        # print(np.max(out))
        out = self.soft.forward(out)
        return out

    def train(self, batch, lr=0.0005):
        out, loss, acc = self.forward(batch)
        gradient = self.soft.backprop(batch[0])
        gradient = self.fcl1.backprop(gradient, lr)
        # gradient = self.pool4.backprop(gradient)
        # gradient = self.conv4.backprop(gradient, lr) 
        # gradient = self.pool3.backprop(gradient)
        # gradient = self.conv3.backprop(gradient, lr) 
        # gradient = self.pool2.backprop(gradient)
        # gradient = self.relu2.backprop(gradient)
        # gradient = self.conv2.backprop(gradient, lr) 
        gradient = self.pool1.backprop(gradient)
        # gradient = self.relu1.backprop(gradient)
        gradient = self.conv1.backprop(gradient, lr)        

        return out, loss, acc
