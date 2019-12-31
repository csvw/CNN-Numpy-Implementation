from ImagePreprocessor import ImagePreprocessor
import numpy as np
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    testfilepath = Path.cwd() / 't_25_4.png'
    resultpath = 'testresult1.png'
    resultpath2 = 'testresult2.png'

    p = ImagePreprocessor()
    im = p.open_image(testfilepath)
    r1 = p.shrink_image(im, 80)
    r2 = p.shrink_image(im, 160)
    r1.save(resultpath)
    r2.save(resultpath2)
