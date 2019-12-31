import csv
from CNN import CNNMR

class ModelLoaderMR:
    def __init__(self, nn):
        self.nn = nn

    def save(self, name):
        self.save_conv(name, self.nn.conv1, 1)
        # self.save_conv(name, self.nn.conv2, 2)
        # self.save_conv(name, self.nn.conv1, 3)
        #self.save_conv(name, self.nn.conv2, 4)
        self.save_fcl(name, self.nn.fcl1, 1)

    def save_fcl(self, fname, fcl, num):
        with open(fname + "_fcl" + str(num) + ".csv", mode='w') as fcl_file:
            fcl_writer = csv.writer(fcl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(fcl.W.shape[0]):
                to_write = []
                for j in range(fcl.W.shape[1]):
                    to_write.append(fcl.W[i, j])
                fcl_writer.writerow(to_write)
            to_write = []
            for j in range(fcl.b.shape[1]):
                to_write.append(fcl.b[0][j])
            fcl_writer.writerow(to_write)

    def save_conv(self, fname, conv, num):
        with open(fname + "_conv_filters" + str(num) + ".csv", mode='w') as filter_file:
            filter_writer = csv.writer(filter_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            for f in conv.F:
                to_write = []
                for i in range(len(f)):
                    for j in range(len(f[0])):
                        for c in range(len(f[0][0])):
                            to_write.append(f[i, j, c])
                filter_writer.writerow(to_write)


    def load(self, name):
        self.load_fcl(name, self.nn.fcl1, 1)
        self.load_conv(name, self.nn.conv1, 1)
        # self.load_conv(name, self.nn.conv2, 2)
        # self.load_conv(name, self.nn.conv3, 3)
        # self.load_conv(name, self.nn.conv4, 4)

    def load_fcl(self, name, fcl, num):
        with open(name + "_fcl" + str(num) + ".csv", mode='r') as fcl_file:
            fcl_reader = csv.reader(fcl_file, delimiter=',')

            matrix_row = 0
            for row in fcl_reader:
                if matrix_row == fcl.W.shape[0]:
                    for idx in range(len(row)):
                        fcl.b[0][idx] = float(row[idx])
                else:
                    for idx in range(len(row)):
                        fcl.W[matrix_row][idx] = float(row[idx])
                matrix_row += 1

    def load_conv(self, name, conv, num):
        with open(name + "_conv_filters" + str(num) + ".csv", mode='r') as filter_file:
            filter_reader = csv.reader(filter_file, delimiter=',')

            f = 0
            for row in filter_reader:
                for idx in range(len(row)):
                    j = idx % 3
                    i = (idx % 9) // 3
                    c = idx // 9
                    conv.F[f][i][j][c] = float(row[idx])
                f += 1
            #print(conv.F[0])

