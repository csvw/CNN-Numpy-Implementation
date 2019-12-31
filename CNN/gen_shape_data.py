import numpy as np
from PIL import Image
from pathlib import Path
import random
import csv

data_directory = Path.cwd() / 'shapedata'

width  = 320
height = 256

def gen_blank_image():
    np_im = np.full(shape=(height, width), fill_value=255)
    return np_im

def draw_rect(np_im, x, y, size):
    for i in range(x, x+size):
        np_im[y][i]      = 201
        np_im[y+size][i] = 201
    for i in range(y, y+size):
        np_im[i][x]      = 201
        np_im[i][x+size] = 201
    return np_im

def rand_rect_dims(x_constraint, y_constraint):
    x    = random.randint(x_constraint-12, x_constraint+12)
    y    = random.randint(y_constraint-12, y_constraint+12)
    size = random.randint(32, 48)
    return (x, y, size)

def draw_rand_rect(np_im, x_constraint, y_constraint):
    x, y, size = rand_rect_dims(x_constraint, y_constraint)
    draw_rect(np_im, x, y, size)
    return (x, y, size)

def draw_rand_rects(np_im, num_rects):
    c = (128, 160)
    labels = []
    for r in range(num_rects):
        west_east  = -80 + int(r > 3) * 120
        left_right = -22 + (r // 2) * 44 if r < 4 else ((r-4)//3) * 44
        row = r % 2 if r < 4 else (r-4) % 3
        north_south = 64 + 44 * row
        x_constraint = c[1] + west_east + left_right
        y_constraint = north_south
        label = draw_rand_rect(np_im, x_constraint, y_constraint)
        labels.append(label)
    return labels

def generate_images():
    img_label_vec = []
    for i in range(500):
        np_im = gen_blank_image()
        img_labels = draw_rand_rects(np_im, 10)
        print(np_im.shape)
        im = Image.fromarray(np_im.astype('uint8')).convert('L')
        im.save('shapedata/t' + str(i) + '.png')
        img_label_vec.append(('shapedata/t' + str(i) + '.png', img_labels))

    save_labels(img_label_vec)


def save_labels(label_vec):
    with open('A1_labels.csv', mode='w') as label_file:
        label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for img_name, label in label_vec:
            to_write = [img_name]
            for i in range(len(label)):
                to_write.append(label[i])
            label_writer.writerow(to_write)

if __name__ == "__main__":
    generate_images()