import numpy as np
from PIL import Image

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
        for b in range(batch.shape[2]):
            for i in range(batch.shape[0] - 2 * self.filter_edge):
                for j in range(batch.shape[1] - 2 * self.filter_edge):
                    for f in range(self.num_filters):
                        yield b, i, j, f

    def compute_new_dims(self, batch):
        dimy, dimx, num_f, batch_size = batch.shape
        dimy -= 2 * self.filter_edge
        dimx -= 2 * self.filter_edge
        return dimy, dimx, num_f, batch_size

    def apply(self, batch):
        self.last_input = batch
        batch = self.pad_batch(batch)

        dimy, dimx, num_f, batch_size = self.compute_new_dims(batch)
        convolution_cube = np.zeros((dimy, dimx, self.num_filters, batch_size))

        for b, i, j, f in self.iterate(batch):  # batch: FXFXC filter: FXFXC
            convolution_cube[i, j, f, b] = np.sum(batch[i:i+self.filter_dims, j:j+self.filter_dims, :, b] * self.filters[f]) + self.bias[f]
        
        # print("Input: " + str(self.last_input.shape))
        # print("Output: " + str(convolution_cube.shape))

        return convolution_cube

    def test_conv_forward(self, batch):
        a = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
        test_filters = np.repeat(a[:, :, np.newaxis], self.num_filters, axis=2)
        print(test_filters.shape)

        dimy, dimx, num_f, batch_size = self.compute_new_dims(batch)
        convolution_cube = np.zeros((dimy, dimx, self.num_filters, batch_size))

        for b, i, j, f in self.iterate(batch):
            convolution_cube[i, j, f, b] = np.sum(batch[i:i+self.filter_dims, j:j+self.filter_dims, b] * test_filters[:, :, f])
        
        print(batch.shape)
        for b in range(batch_size):
            im = Image.fromarray(((convolution_cube[:, :, 0, b])).astype(np.uint8))
            im.save('test_CNN/test_conv_forward' + str(b) + '.png')
            im = Image.fromarray(((batch[:, :, b])).astype(np.uint8))
            im.save('test_CNN/test_conv_forwardimg' + str(b) + '.png')

    def pad_batch(self, batch):
        batch = np.pad(batch, ((1, 1), (1, 1), (0, 0), (0,0)), 'constant')
        #print(batch.shape)
        return batch

    def backprop(self, dLoss_dOutput, learn_rate):
        dLoss_dFilters = np.zeros(self.filters.shape)
        dLoss_dBias = np.zeros(self.bias.shape)
        dLoss_dInput = np.zeros(self.last_input.shape)

        dimy, dimx, num_f, batch_size = self.last_input.shape

        # print(dLoss_dFilters.shape)
        # print(dLoss_dOutput.shape)

        for i in range(dimy - 2 * self.filter_edge):
            for j in range(dimx - 2 * self.filter_edge):
                for f in range(self.num_filters):
                    region_batch = self.last_input[i:i+self.filter_dims, j:j+self.filter_dims, :]
                    for b in range(batch_size): # DLDI: WXHXFXB DLDO: WXHXCXB  Filter by DLDO: (WXHXC) * (1)
                        region = region_batch[:, :, :, b] # for every filter slice (wXhXc) += (wxhxc) * 1
                        dLoss_dFilters[f] += dLoss_dOutput[i, j, f, b] * region / batch_size
                        dLoss_dBias[f] += dLoss_dOutput[i, j, f, b] / batch_size
                        dLoss_dInput[i:i+self.filter_dims, j:j+self.filter_dims, :, b] += self.filters[f] * dLoss_dOutput[i, j, f, b]

        self.filters -= learn_rate * dLoss_dFilters
        self.bias    -= learn_rate * dLoss_dBias

        #print(self.filters)

        return dLoss_dInput