import numpy as np

class Convolution:
    def __init__(self, dims, num_filters):
        self.filter_dims = dims
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, dims, dims) / (dims ** 2)
        self.filter_edge = int(dims/2)
        self.last_input = 0
        self.last_batch = 0

    def apply(self, image):
        self.last_input = image
        dimy = image.shape[0] - 2 * self.filter_edge
        dimx = image.shape[1] - 2 * self.filter_edge
        convolution_cube = np.zeros((dimy, dimx, self.num_filters))

        for i in range(image.shape[0] - 2 * self.filter_edge):
            for j in range(image.shape[1] - 2 * self.filter_edge):
                for f in range(self.num_filters):
                    convolution_cube[i, j, f] = np.sum(image[i:i+self.filter_dims, j:j+self.filter_dims] * self.filters[f])
        
        return convolution_cube

    def apply_batch(self, batch):
        self.last_input = batch

        dimy = batch.shape[1] - 2 * self.filter_edge
        dimx = batch.shape[2] - 2 * self.filter_edge
        convolution_cube = np.zeros((batch.shape[0], dimy, dimx, self.num_filters))

        for b in range(batch.shape[0]):
            for i in range(batch.shape[1] - 2 * self.filter_edge):
                for j in range(batch.shape[2] - 2 * self.filter_edge):
                    for f in range(self.num_filters):
                        convolution_cube[b, i, j, f] = np.sum(batch[b, i:i+self.filter_dims, j:j+self.filter_dims] * self.filters[f])
        
        return convolution_cube

    def backprop_batch(self, dLoss_dOutput, learn_rate):
        dLoss_dFilters = np.zeros(self.filters.shape)

        batch_size, dimy, dimx = self.last_input.shape

        for i in range(dimy - 2 * self.filter_edge):
            for j in range(dimx - 2 * self.filter_edge):
                for f in range(self.num_filters):
                    region_batch = self.last_input[:, i:i+3, j:j+3]
                    for b in range(batch_size):
                        region = region_batch[b]
                        dLoss_dFilters[f] += dLoss_dOutput[b, i, j, f] * region

        self.filters -= learn_rate * dLoss_dFilters

        return None

    def backprop(self, dLoss_dOutput, learn_rate):
        dLoss_dFilters = np.zeros(self.filters.shape)

        dimy, dimx = self.last_input.shape

        for i in range(dimy - 2 * self.filter_edge):
            for j in range(dimx - 2 * self.filter_edge):
                for f in range(self.num_filters):
                    region = self.last_input[i:i+3, j:j+3]
                    dLoss_dFilters[f] += dLoss_dOutput[i, j, f] * region

        self.filters -= learn_rate * dLoss_dFilters

        return None