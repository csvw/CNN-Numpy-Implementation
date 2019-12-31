from PIL import Image
import numpy as np
from pathlib import Path
from ShapeArtist import ShapeArtist
import random

class Shapes:
    def __init__(self):
        self.artist = ShapeArtist()
        self.path = Path.cwd() / "DLData" / "shape_classes"
        self.data   = []
        for f in self.path.iterdir():
            name = f.name.split('.')[0].split('_')[0]
            shape = self.artist.name_to_shape[name]
            num = self.artist.shape_numbers[shape]
            label = [1. if i == num else 0. for i in range(len(self.artist.all_shapes))]
            # print(name, num, label)
            im = np.array(Image.open(f))
            batch = self.normalize_batch([label, im])
            self.data.append(batch)
            # print(f.name, label)
        random.shuffle(self.data)

    def normalize_batch(self, batch):
        batch[1] = (batch[1]/255.) - 0.5
        # batch[1] += np.random.normal(0, 0.05, batch[1].shape)
        if batch[1].ndim != 3:
            batch[1] = batch[1][:, :, np.newaxis]
        # print(batch[1].shape)
        return batch