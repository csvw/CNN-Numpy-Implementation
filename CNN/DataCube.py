import numpy as np
from PIL import Image
from pathlib import Path
from ImagePreprocessor import ImagePreprocessor

class DataCube:
    def __init__(self, datapath):
        self.datapath = Path(datapath)
        self.train_images = []
        self.train_labels = []
        self.string_labels = []
        self.label_map = {}
        self.label_set = []
        self.preprocessor = ImagePreprocessor()
        self.load_data()

    def load_data(self):
        for f in self.datapath.iterdir():
            if not ".png" in f.name:
                continue
            im    = Image.open(f)
            im    = self.preprocessor.process_cell_image(im)
            np_im = np.array(im)
            self.train_images.append(np_im)
            label = f.name.split('_')[0]
            self.string_labels.append(label)
        s = set(self.string_labels)
        self.label_set = list(s)
        self.label_map = {self.label_set[i]:i for i in range(len(self.label_set))}
        for i in range(len(self.string_labels)):
            self.train_labels.append(self.label_map[self.string_labels[i]])
