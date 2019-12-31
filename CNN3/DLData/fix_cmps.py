import ImagePreprocessor
from pathlib import Path
from PIL import Image

p = ImagePreprocessor.ImagePreprocessor()

path = Path.cwd() / 'comparisons2'

for f in path.iterdir():
    im = Image.open(f)
    im = p.shrink_image(im, 48)
    im.save('comparisons2/' + f.name)