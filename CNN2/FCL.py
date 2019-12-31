import numpy as np
import math
from PIL import Image

class FCL:
    def __init__(self, m, n):
        self.weights = np.random.randn(n, m) / (m*n)
        self.biases = np.zeros(n)
        self.last_input = 0
        self.last_input_shape = 0
        self.beta = np.zeros(n)
        self.gamma = np.random.randn(n) / n
        self.last_z_norm = 0
        self.avg = 0
        self.var_recip = 0
        self.running_avg = np.zeros(n)
        self.running_var = np.zeros(n)
        self.running_var_recip = np.zeros(n)
        self.running_avg_old = np.zeros(n)
        self.running_var_old = np.zeros(n)
        self.num_values = 0
        self.last_z_mu = 0

    # Citation: (Also found in Knuth's Art of Programming)
    # https://www.johndcook.com/blog/standard_deviation/
    def calc_running(self, x):
        self.num_values += 1
        if self.num_values == 1:
            self.running_avg_old = x
            self.running_avg = x
        else:
            self.running_avg = self.running_avg_old + (x - self.running_avg_old) / self.num_values
            self.running_var = self.running_var_old + (x - self.running_avg_old)*(x - self.running_avg) / (self.num_values - 1)
            self.running_avg_old = self.running_avg
            self.running_var_old = self.running_var
            self.running_var_recip = 1. / np.sqrt(self.running_var + 0.000001)

    def apply_inference(self, input_tensor):
        self.last_input_shape = input_tensor.shape
        input_vector = input_tensor.flatten()
        self.last_input = input_vector
        z = self.weights @ input_tensor
        return (z - self.running_avg) / self.running_var_recip

    def apply(self, input_tensor):
        '''Two cases potentially here:
            N x B input
            W X H X F X B input
            Flattened vs Convolution Cube
            Weights: (Out X N)
            In_Flat: (N X B)
            X: (Out X B)
        '''

        self.last_input_shape = input_tensor.shape
        self.last_input = input_tensor
        input_flat = input_tensor

        if input_tensor.ndim > 2:
            input_flat = np.zeros((self.weights.shape[1], input_tensor.shape[-1]))
            for b in range(input_tensor.shape[-1]):
                input_flat[:, b] = input_tensor[:, :, :, b].flatten()
            self.last_input = input_flat

        # print("forward: tensor flat weights")
        # print(input_tensor.shape)
        # print(input_flat.shape)
        # print(self.weights.shape)

        z = self.weights @ input_flat + self.biases[:, np.newaxis]

        return z

    # def test_fcl_forward(self, input_tensor):
    #     if input_tensor.ndim > 2:
    #         input_flat = np.zeros((self.weights.shape[1], input_tensor.shape[0]))
    #         self.last_input = input_flat
    #         for b in range(input_tensor.shape[-1]):
    #             input_flat[:, b] = input_tensor[:, :, :, b].flatten()
    #     im = Image.fromarray(input_flat.reshape(input_tensor.shape)[:, :, 0, 0])
    #     im.save('test_CNN/test_fcl_forward.png')

    # def batch_norm(self, z):
    #     #print(z.shape)
    #     for b in range(z.shape[0]):
    #         self.calc_running(z[b])
    #     avg = np.sum(z, axis=0) / z.shape[0]
    #     var = np.sum((z - avg)**2, axis=0) / z.shape[0]
    #     var_recip = 1. / np.sqrt(var + 0.0001)
    #     self.var_recip = var_recip
    #     self.avg = avg
    #     self.var = var
    #     z_mu = (z - self.avg)
    #     z_norm =  z_mu * self.var_recip
    #     self.last_z_mu
    #     self.last_z_norm = z_norm
    #     z_tild = self.gamma * z_norm + self.beta
    #     return z_tild

    # Citation: Derivatives and backprop algorithm obtained from
    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    #dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
    #- (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
    # def backprop_znorm_batch(self, dLoss_dOutput, learn_rate):
    #     batch_size = self.last_input_shape[0]
    #     #dOutput_dZnorm = dLoss_dOutput * self.gamma
    #     # print(self.last_z_norm.shape)
    #     Nr = 1. / batch_size

    #     dOutput_dInput = Nr * self.gamma * self.var_recip * (batch_size * dLoss_dOutput - np.sum(dLoss_dOutput, axis=0) 
    #                      - self.last_z_mu * (self.var)**(-1.) * np.sum(dLoss_dOutput * self.last_z_mu, axis=0))
    #     dOutput_dBeta = np.sum(dLoss_dOutput, axis=0)
    #     dOutput_dGamma  = np.sum(self.last_z_norm * dLoss_dOutput, axis=0)

    #     #print(dOutput_dGamma.shape)

    #     self.gamma -= learn_rate * dOutput_dGamma
    #     self.beta  -= learn_rate * dOutput_dBeta
        

    #     return dOutput_dInput

    # Citation: Derivatives and backprop algorithm obtained from
    # https://deepnotes.io/batchnorm
    # def backprop_znorm_batch(self, dLoss_dOutput, learn_rate):
    #     batch_size = self.last_input_shape[0]
    #     dOutput_dZnorm = dLoss_dOutput * self.gamma
    #     print("back")
    #     print(self.last_z_norm.shape)
    #     print(dOutput_dZnorm.shape)
        
    #     dOutput_dVar = np.sum(dOutput_dZnorm * self.avg, axis=0) * -0.5 * (self.var + 0.00001)**(-3./2)
    #     dOutput_dAvg = np.sum(dOutput_dZnorm * -self.var_recip, axis=0) + dOutput_dVar * (1./batch_size) * np.sum(-2. * self.avg, axis=0)
    #     dOutput_dInput = (dOutput_dZnorm * self.var_recip) + (dOutput_dAvg / batch_size) + (dOutput_dVar * 2./batch_size * self.avg)
    #     dOutput_dBeta = np.sum(dLoss_dOutput, axis=0)
    #     dOutput_dGamma  = np.sum(self.last_z_norm * dLoss_dOutput, axis=0)

    #     self.gamma -= learn_rate * dOutput_dGamma
    #     self.beta  -= learn_rate * dOutput_dBeta

    #     print(dOutput_dInput.shape)

    #     return dOutput_dInput


    # Citation: backprop formula: DL/DW = DL/DY @ X.T obtained from
    # https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/

    def backprop(self, dLoss_dOutput, learn_rate):
        dOutput_dInput   = self.weights
        dOutput_dWeights = self.last_input

        # print("Backprop")
        # print(dOutput_dWeights.shape)
        # print(dLoss_dOutput.shape)

        dLoss_dWeights   = dLoss_dOutput @ dOutput_dWeights.T / self.last_input_shape[-1]
        dLoss_dInput     = dOutput_dInput.T @ dLoss_dOutput
        dLoss_dBias      = np.sum(dLoss_dOutput, axis=1) / self.last_input_shape[-1]

        # print(dLoss_dWeights.shape)
        # print(dLoss_dBias.shape)

        self.biases  -= learn_rate * dLoss_dBias.reshape(self.biases.shape)
        self.weights -= learn_rate * dLoss_dWeights

        return dLoss_dInput.reshape(self.last_input_shape)

