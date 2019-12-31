from PIL import Image
import numpy as np
from pathlib import Path
from ShapeArtist import ShapeArtist

class Shapes:
    def __init__(self):
        self.artist = ShapeArtist()
        self.path = Path.cwd() / "DLData/shape_classes"
        self.data   = []
        for f in path.iterdir():
            name = f.name.split('.')
            shape = self.artist.name_to_shape[name]
            num = self.artist.shape_numbers[shape]
            label = [1 if i == num else 0 for i in range(len(self.artist.all_shapes))]
            self.data.append((label, np.array(Image.open(f))))