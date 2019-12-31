import numpy as np

class Maxpool:
    def __init__(self):
        self.last_input = 0

    def apply(self, convolution_cube):
        self.last_input = convolution_cube
        dimy, dimx, num_filters = convolution_cube.shape
        half_dimy = int(dimy / 2)
        half_dimx = int(dimx / 2)

        out = np.zeros((half_dimy, half_dimx, num_filters))

        for i in range(half_dimy):
            for j in range(half_dimx):
                out[i, j] = np.amax(convolution_cube[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)], axis = (0, 1))

        return out

    def apply_batch(self, convolution_cube):
        self.last_input = convolution_cube
        batch_size, dimy, dimx, num_filters = convolution_cube.shape
        half_dimy = int(dimy / 2)
        half_dimx = int(dimx / 2)

        out = np.zeros((batch_size, half_dimy, half_dimx, num_filters))

        for b in range(batch_size):
            for i in range(half_dimy):
                for j in range(half_dimx):
                    out[:, i, j, :] = np.amax(convolution_cube[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2), :], axis = (1, 2))

        return out

    def backprop_batch(self, dLoss_dOutput):
        dLoss_dInput = np.zeros(self.last_input.shape)
        batch_size, dimy, dimx, numf = self.last_input.shape
        half_dimy = dimy // 2
        half_dimx = dimx // 2

        #print(self.last_input.shape)

        for i in range(half_dimy):
            for j in range(half_dimx):
                region_batch = self.last_input[:, (i*2):(i*2+2), (j*2):(j*2+2)]
                for b in range(batch_size):
                    region = region_batch[b]
                    max_vec = np.amax(region, axis=(1,2))
                    for i2 in range(2):
                        for j2 in range(2):
                            for f2 in range(numf):
                                if region[i2, j2, f2] == max_vec[f2]:
                                    dLoss_dInput[b, i*2+i2, j*2+j2,f2] = dLoss_dOutput[i, j, f2]

        return dLoss_dInput

    # def backprop_batch(self, dLoss_dOutput):
    #     dLoss_dInput = np.zeros(self.last_input.shape[1:])
    #     batch_size, dimy, dimx, numf = self.last_input.shape
    #     half_dimy = dimy // 2
    #     half_dimx = dimx // 2

    #     #print(self.last_input.shape)

    #     last_input_avg = np.sum(self.last_input, axis=0) / batch_size

    #     for i in range(half_dimy):
    #         for j in range(half_dimx):
    #             region = last_input_avg[(i*2):(i*2+2), (j*2):(j*2+2)]
    #             max_vec = np.amax(region, axis=(0,1))
    #             for i2 in range(2):
    #                 for j2 in range(2):
    #                     for f2 in range(numf):
    #                         if region[i2, j2, f2] == max_vec[f2]:
    #                             dLoss_dInput[i*2+i2, j*2+j2,f2] += dLoss_dOutput[i, j, f2]

    #     return dLoss_dInput

    def backprop(self, dLoss_dOutput):
        '''
        Blow the image back up to its original size, and paste the max pixels back onto their original locations.
        The other pixels are all left black.
        '''
        dLoss_dInput = np.zeros(self.last_input.shape)
        dimy, dimx, numf = self.last_input.shape
        half_dimy = dimy // 2
        half_dimx = dimx // 2

        for i in range(half_dimy):
            for j in range(half_dimx):
                region = self.last_input[(i*2):(i*2+2), (j*2):(j*2+2)]
                max_vec = np.amax(region, axis=(0,1))
                for i2 in range(2):
                    for j2 in range(2):
                        for f2 in range(numf):
                            if region[i2, j2, f2] == max_vec[f2]:
                                dLoss_dInput[i*2+i2, j*2+j2,f2] = dLoss_dOutput[i, j, f2]

        return dLoss_dInput
