import numpy as np
from PIL import Image

class Pool:
    def __init__(self):
        self.last_input = 0

    def apply(self, convolution_cube):
        self.last_input = convolution_cube
        dimy, dimx, num_filters, batch_size = convolution_cube.shape
        half_dimy = int(dimy / 2)
        half_dimx = int(dimx / 2)

        out = np.zeros((half_dimy, half_dimx, num_filters, batch_size))

        for i in range(half_dimy):
            for j in range(half_dimx):
                out[i, j] = np.amax(convolution_cube[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)], axis = (0, 1))

        return out

    def test_pool_forward(self, convolution_cube):
        dimy, dimx, num_filters, batch_size = convolution_cube.shape
        half_dimy = int(dimy / 2)
        half_dimx = int(dimx / 2)

        out = np.zeros((half_dimy, half_dimx, num_filters, batch_size))

        for b in range(batch_size):
            for i in range(half_dimy):
                for j in range(half_dimx):
                    out[i, j] = np.amax(convolution_cube[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)], axis = (0, 1))

        for b in range(batch_size):
            im = Image.fromarray(out[:, :, 0, b].astype(np.uint8))
            im.save('test_CNN/test_pool_forward' + str(b) + '.png')

    def backprop(self, dLoss_dOutput):
        dLoss_dInput = np.zeros(self.last_input.shape)
        dimy, dimx, numf, batch_size = self.last_input.shape
        half_dimy = dimy // 2
        half_dimx = dimx // 2

        #print(self.last_input.shape)

        for i in range(half_dimy):
            for j in range(half_dimx):
                region_batch = self.last_input[(i*2):(i*2+2), (j*2):(j*2+2)]
                for b in range(batch_size):
                    region = region_batch[:, :, :, b]
                    max_vec = np.amax(region, axis=(0,1))
                    for i2 in range(2):
                        for j2 in range(2):
                            for f2 in range(numf):
                                if region[i2, j2, f2] == max_vec[f2]:
                                    dLoss_dInput[i*2+i2, j*2+j2, f2, b] = dLoss_dOutput[i, j, f2, b]

        return dLoss_dInput
