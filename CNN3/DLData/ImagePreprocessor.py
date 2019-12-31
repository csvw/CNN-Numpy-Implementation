import numpy as np
from PIL import Image
from pathlib import Path

class ImagePreprocessor:
    def convert_grayscale(self, image):
        try:
            if isinstance(image, str):
                image_path = Path(image)
                im = Image.open(image_path)
            elif isinstance(image, Path):
                image_path = image
                im = Image.open(image_path)
            else:
                im = image
            gray_im = im.convert('L')
            return gray_im
        except IOError:
            print("Unable to convert " + image_path.name + " to grayscale.")

    def invert_image(self, image):
        blank_im = np.array([0 for i in range(len(image.shape[0]))] for j in range(len(image.shape[1])))
        im = np.min((image -  255) * -1, blank_im)
        return im

    def maxpool(self, numpy_image_array):
        h, w = numpy_image_array.shape
        half_h = h // 2
        half_w = w // 2
        quarter_image = np.zeros((half_h, half_w))
        for i in range(half_h):
            for j in range(half_w):
                quarter_image[i, j] = np.amax(numpy_image_array[(i*2):(i*2+2), (j*2):(j*2+2)])
        return quarter_image

    def shrink_image(self, im, target_width=320):
       scaling_ratio = target_width/float(im.size[0])
       target_height = int(float(im.size[1]) * float(scaling_ratio))
       result = im.resize((target_width, target_height), Image.ANTIALIAS)
       return result

    def open_image(self, image):
        if isinstance(image, str):
            image_path = Path(image)
            im = Image.open(image_path)
        elif isinstance(image, Path):
            image_path = image
            im = Image.open(image_path)
        else:
            im = image
        return im

    def process_problem_image(self, image):
        im = self.open_image(image)
        im = self.shrink_image(im).convert("L")
        return im

    def process_cell_image(self, image):
        im = self.open_image(image)
        im = self.shrink_image(im, 32).convert("L")
        return im
