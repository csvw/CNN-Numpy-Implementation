import numpy as np
from PIL import Image
from pathlib import Path
from ImagePreprocessor import ImagePreprocessor
import csv

datapath = '/home/shae/Code/CNN/shapedata'
csvpath = '/home/shae/Code/CNN/A1_labels.csv'

class SquareData:
    def __init__(self):
        self.datapath = Path(datapath)
        self.csvpath = Path(csvpath)
        self.keys = []
        self.train_images = {}
        self.train_labels = {}
        self.load_data()

    def load_data(self):
        with open(csvpath, mode='r') as label_file:
            csv_reader = csv.reader(label_file, delimiter=',')
            for row in csv_reader:
                vals = []
                for x in range(1, 11):
                    for val in eval(row[x]):
                        vals.append(val)
                self.train_labels[row[0]] = np.array(vals)

        for f in self.datapath.iterdir():
            self.keys.append('shapedata/' + f.name)
            im    = Image.open(f)
            np_im = np.array(im)
            self.train_images['shapedata/' + f.name] = np_im

        for key in self.keys:
            test = self.train_labels[key]