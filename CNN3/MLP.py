import numpy as np
from Conv import Conv
from Pool import Pool
from Relu import Relu
from FCL import FCL
from Soft import Soft

class MLP:
    def __init__(self):
        self.relu1 = Relu()
        self.fcl1  = FCL(49 * 49, 1000)
        # self.fcl1  = FCL(28 * 28, 1000)
        self.relu2 = Relu()
        self.fcl2  = FCL(1000, 500)
        self.fcl3  = FCL(500, 10)
        # self.fcl3 = FCL(500, 10)
        self.soft  = Soft()

    def normalize_batch(self, batch):
        batch[1] = (batch[1]/255.) - 0.5
        # batch[1] += np.random.normal(0, 0.05, batch[1].shape)
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
        out = self.fcl1.forward(batch)
        out = self.relu1.forward(out)
        out = self.fcl2.forward(out)
        out = self.relu2.forward(out)
        out = self.fcl3.forward(out)
        out = self.soft.forward(out)
        return out

    def train(self, batch, lr=0.0005):
        out, loss, acc = self.forward(batch)
        gradient = self.soft.backprop(batch[0])
        gradient = self.fcl3.backprop(gradient, lr)
        gradient = self.relu2.backprop(gradient)
        gradient = self.fcl2.backprop(gradient, lr)
        gradient = self.relu1.backprop(gradient)
        gradient = self.fcl1.backprop(gradient, lr)       

        return out, loss, acc
