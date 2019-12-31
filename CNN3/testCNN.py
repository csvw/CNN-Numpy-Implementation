from CNN import CNNMR
from ModelLoader2 import ModelLoaderMR
import ShapeData
import numpy as np
import mnist
from PIL import Image

if __name__ == "__main__":
    nn = CNNMR()
    md_loader = ModelLoaderMR(nn)
    md_loader.load('test_nan')
    train = mnist.train_images()
    label = mnist.train_labels()
    for j in range(1):
        for i, (lab, tra) in enumerate(zip(label, train)):
            im = Image.fromarray(tra).convert("L")
            l = [1 if i == lab else 0 for i in range(10)]
            batch = [l, tra]
            #nn.test(batch)
            p, _, _ = nn.forward(batch)
            print(lab)
            print(p)
            im.save(str(i) + ".png")
            input("Press enter to continue...")
                
