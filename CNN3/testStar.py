import ShapeArtist
from PIL import Image

if __name__ == "__main__":
    a = ShapeArtist.ShapeArtist()
    im = a.gen_blank_image()
    a.star(im, 24, 24, 30, 1)
    im = Image.fromarray(im)
    im.save("star.png")
    im = a.gen_blank_image()
    a.hexagon(im, 24, 24, 24, 1)
    im = Image.fromarray(im)
    im.save("hexagon.png")
    im = a.gen_blank_image()
    a.octagon(im, 24, 24, 30, 1)
    im = Image.fromarray(im)
    im.save("octagon.png")