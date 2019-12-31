import numpy as np
from PIL import Image
from pathlib import Path
import math
from ImagePreprocessor import ImagePreprocessor
import random
from ShapeArtist import ShapeArtist
from PIL import ImageFilter

class Generator:
    def __init__(self):
        self.artist = ShapeArtist()

    def gen_shape_data(self):
        for shape in self.artist.all_shapes:
            for sz in range(18, 38, 1):
                for fill in range(-1, 2):
                    for rotation in range(0, 360, 90):
                        for transx in range(-6, 7, 2):
                            for transy in range(-6, 7, 2):
                                im = self.artist.gen_blank_image()
                                shape(im, 24 + transx, 24 + transy, sz, fill)
                                im = self.rotate(im, rotation)
                                im = Image.fromarray(im).convert('L')
                                #if tf != 0:
                                    #im = im.filter(ImageFilter.GaussianBlur(tf))
                                im.save('shape_classes/' + self.artist.shape_names[shape] + '_' + str(sz) + str(fill+1) + str(rotation) + str(transx) + str(transy) + '.png')
    
    def rotate(self, np_im, rotation):
        tmp = Image.fromarray(self.artist.gen_blank_image(99))
        im = Image.fromarray(np_im).convert('L')
        tmp.paste(im, (25, 25))
        rot = tmp.rotate(rotation, expand=False)
        # x = int((rotation % 180) / 100) * 24
        rot.save("rot_test.png")
        region = rot.crop((25, 25, 74, 74))
        return np.array(region)

    def translate(self, np_im, transX, transY):
        tmp = Image.fromarray(self.artist.gen_blank_image(99))
        im = Image.fromarray(np_im).convert('L')
        tmp.paste(im, (25, 25))
        # rot = tmp.rotate(rotation, expand=False)
        # x = int((rotation % 180) / 100) * 24
        region = rot.crop((25, 25, 74, 74))
        return np.array(region)

if __name__ == "__main__":
    g = Generator()
    g.gen_shape_data()
