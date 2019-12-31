from PIL import Image
import numpy as np
from pathlib import Path
from ImagePreprocessor import ImagePreprocessor
from ExtractShapes import Extractor
from CNN import CNNMR
from ModelLoader2 import ModelLoaderMR
from MLP import MLP
from MLPLoader import MLPLoader

class ShapeData:
    def __init__(self):
        self.e = 0
        self.w = 9999
        self.n = 9999
        self.s = 0
        self.center = (0, 0)
        self.area = 0
        self.shape = 0

class Analyst:
    def __init__(self, im):
        self.nn = MLP()
        self.model_loader = MLPLoader(self.nn)
        self.model_loader.load('shape_classifier_5')
        self.p = ImagePreprocessor()
        self.extractor = Extractor(im)
        self.regions = self.extractor.shapes
        self.shapes = []
        self.init_shapes()
        self.num_shapes = len(self.shapes)
        self.display()
    
    def init_shapes(self):
        for region in self.regions:
            d = ShapeData()
            for node in region:
                n = node.loc
                if n[0] > d.e:
                    d.e = n[0]
                if n[0] < d.w:
                    d.w = n[0]
                if n[1] > d.s:
                    d.s = n[1]
                if n[1] < d.n:
                    d.n = n[1]
            d.area = (d.s - d.n) * (d.e - d.w)
            d.center = ((d.e + d.w) // 2, (d.n + d.s) // 2)
            d.shape = self.classify(region)
            self.shapes.append(d)

    def classify(self, region):
        dims = self.extractor.im.shape
        im = np.full_like(self.extractor.im, 255)
        for n in region:
            i = n.loc[0]
            j = n.loc[1]
            im[i][j] = 0
        im = Image.fromarray(im)
        im = self.p.shrink_image(im, 49)
        im = np.array(im)
        im = self.normalize(im)
        out = self.nn.apply(im)
        return np.argmax(out)   

    def normalize(self, batch):
        batch = (batch/255.) - 0.5
        if batch.ndim != 3:
            batch = batch[:, :, np.newaxis]
        # print(batch[1].shape)
        return batch

    def display(self):
        for d in self.shapes:
            print(d.e, d.w, d.n, d.s)
            print(d.center)
            print(d.area)
            print(d.shape)

if __name__ == '__main__':
    path = Path.cwd() / 'a.png'
    im = Image.open(path).convert('L')
    analyst = Analyst(im)