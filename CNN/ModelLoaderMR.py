import csv
from NeuralNetworkMultipleRegression import NeuralNetworkMR

class ModelLoaderMR:
    def __init__(self, nn):
        self.nn = nn

    def preprocess_name(self, fname):
        if not ".csv" in fname:
            return fname
        return fname.split(".")[0]

    def save_model(self, fname):
        fname = self.preprocess_name(fname)
        with open(fname + "_conv_filters.csv", mode='w') as filter_file:
            filter_writer = csv.writer(filter_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            for f in self.nn.convolution.filters:
                print(f)
                to_write = []
                for i in range(len(f)):
                    for j in range(len(f[0])):
                        to_write.append(f[i, j])
                filter_writer.writerow(to_write)

        with open(fname + "_fcl.csv", mode='w') as fcl_file:
            fcl_writer = csv.writer(fcl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(self.nn.fcl.weights.shape[0]):
                to_write = []
                for j in range(self.nn.fcl.weights.shape[1]):
                    to_write.append(self.nn.fcl.weights[i, j])
                fcl_writer.writerow(to_write)
            
            fcl_writer.writerow([b for b in self.nn.fcl.biases])

        with open(fname + "_reg.csv", mode='w') as fcl_file:
            fcl_writer = csv.writer(fcl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(self.nn.regression.fcl.weights.shape[0]):
                to_write = []
                for j in range(self.nn.regression.fcl.weights.shape[1]):
                    to_write.append(self.nn.regression.fcl.weights[i, j])
                fcl_writer.writerow(to_write)
            
            fcl_writer.writerow([b for b in self.nn.regression.fcl.biases])
    
    def load_model(self, fname):
        fname = self.preprocess_name(fname)
        with open(fname + "_conv_filters.csv", mode='r') as filter_file:
            filter_reader = csv.reader(filter_file, delimiter=',')

            f = 0
            for row in filter_reader:
                for idx in range(len(row)):
                    j = idx % 3
                    i = idx // 3
                    self.nn.convolution.filters[f][i][j] = float(row[idx])
                f += 1
        
        with open(fname + "_fcl.csv", mode='r') as fcl_file:
            fcl_reader = csv.reader(fcl_file, delimiter=',')

            matrix_row = 0
            for row in fcl_reader:
                if matrix_row == self.nn.fcl.weights.shape[0]:
                    for idx in range(len(row)):
                        self.nn.fcl.biases[idx] = float(row[idx])
                else:
                    for idx in range(len(row)):
                        self.nn.fcl.weights[matrix_row][idx] = float(row[idx])
                    matrix_row += 1

        with open(fname + "_reg.csv", mode='r') as fcl_file:
            fcl_reader = csv.reader(fcl_file, delimiter=',')

            matrix_row = 0
            for row in fcl_reader:
                if matrix_row == self.nn.regression.fcl.weights.shape[0]:
                    for idx in range(len(row)):
                        self.nn.regression.fcl.biases[idx] = float(row[idx])
                else:
                    for idx in range(len(row)):
                        self.nn.regression.fcl.weights[matrix_row][idx] = float(row[idx])
                    matrix_row += 1