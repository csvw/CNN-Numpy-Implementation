import numpy as np
from PIL import Image
from pathlib import Path
import math
from ImagePreprocessor import ImagePreprocessor
import random

class DataGenerator:
    def __init__(self):
        self.image_processor = ImagePreprocessor()
        self.partial_rotation_shapes = [self.square, self.diamond, self.plus]
        self.all_shapes = []
        self.full_rotation_shapes = [self.pacman, self.triangle, self.heart, self.right_triangle]
        self.reflection_shapes = [self.pacman, self.right_triangle]
        self.init_all_shapes()

    def init_all_shapes(self):
        self.all_shapes.append(self.circle)
        self.all_shapes.append(self.square)
        self.all_shapes.append(self.heart)
        self.all_shapes.append(self.pacman)
        self.all_shapes.append(self.triangle)
        self.all_shapes.append(self.right_triangle)
        self.all_shapes.append(self.plus)

    def gen_blank_image(self, sz=49):
        np_im = np.full(shape=(sz, sz), fill_value=255)
        np_im = np.uint8(np_im)
        return np_im

    def iterator(self, sz=49):
        for i in range(sz-1):
            for j in range(sz-1):
                yield i, j

    def fill_pixel(self, im, x, y, fill_type=-1):
        if fill_type == 0:
            im[y][x] = 0
        elif fill_type == 1:
            im[y][x] = 124
        else:
            im[y][x] = 255

    def circle(self, im, x, y, sz, fill_type=-1):
        r = sz // 2
        for i, j in self.iterator(im.shape[0]):
            val = (x-i)**2 + (y-j)**2
            if val < (r**2 + r) and val > (r**2 - r):
                self.fill_pixel(im, i, j, 0)
            elif val < r**2:
                self.fill_pixel(im, i, j, fill_type)
    
    def square(self, im, x, y, sz, fill_type=-1):
        for i, j in self.iterator(im.shape[0]):
            r  = (x + (sz // 2)) == i
            l  = (x - (sz // 2)) == i
            b  = (y + (sz // 2)) == j
            t  = (y - (sz // 2)) == j
            lr = abs(i - x) < (sz // 2)
            tb = abs(j - y) < (sz // 2)
            olr = abs(i - x) > (sz // 2)
            otb = abs(j - y) > (sz // 2)
            if (r or l or t or b) and not (olr or otb):
                self.fill_pixel(im, i, j, 0)
            elif lr and tb:
                self.fill_pixel(im, i, j, fill_type)

    def triangle(self, im, x, y, sz, fill_type=-1):
        lp = (x - (sz // 2), (y + (sz // 2)))
        rp = (x + (sz // 2), (y + (sz // 2)))
        tp = (x, (y - (sz // 2)))
        for i, j in self.iterator(im.shape[0]):
            lp_tp_s = (tp[1] - lp[1]) / (tp[0] - lp[0])
            rp_tp_s = (tp[1] - rp[1]) / (tp[0] - rp[0])
            l1 = j - tp[1] - lp_tp_s * (i - tp[0])
            l2 = j - tp[1] - rp_tp_s * (i - tp[0])
            lp_tp = abs(l1) < 2
            rp_tp = abs(l2) < 2
            b = j == rp[1]
            in_bounds = (j > tp[1]) and (j <= rp[1]) and(i > lp[0]) and(i < rp[0])
            if (lp_tp or rp_tp or b) and in_bounds:
                self.fill_pixel(im, i, j, 0)
            if(l1 > 0 and l2 > 0 and not b and in_bounds):
                self.fill_pixel(im, i, j, fill_type)

    def slope(self, p1, p2):
        return (p2[1] - p1[1]) / (p2[0] - p1[0])

    def diamond(self, im, x, y, sz, fill_type=-1):
        r = sz // 2
        n = (x, y - r)
        s = (x, y + r)
        w = (x - r, y)
        e = (x + r, y)
        nws = self.slope(n, w)
        nes = self.slope(n, e)
        for i, j in self.iterator(im.shape[0]):
            l1 = j - n[1] - nws * (i - n[0])
            l2 = j - s[1] - nws * (i - s[0])
            l3 = j - n[1] - nes * (i - n[0])
            l4 = j - s[1] - nes * (i - s[0])
            e1 = abs(l1) < 1.
            e2 = abs(l2) < 1.
            e3 = abs(l3) < 1.
            e4 = abs(l4) < 1.
            in_bounds = (j <= s[1]) and (j >= n[1]) and (i >= w[0]) and (i <= e[0])
            if (e1 or e2 or e3 or e4) and in_bounds:
                self.fill_pixel(im, i, j, 0)
        self.fill(im, x, y, im.shape[0], fill_type)

    def right_triangle(self, im, x, y, sz, fill_type=-1):
        for i, j in self.iterator(im.shape[0]):
            r  = (x + (sz // 2)) == i
            b  = (y + (sz // 2)) == j
            d  = (x - i) + (y - j) == 0
            br = ((x - i) + (y - j) < 0) and (i < (x + (sz // 2))) and (j < (y + (sz // 2)))
            olr = abs(i - x) > (sz // 2)
            otb = abs(j - y) > (sz // 2)
            if (r or b or d) and not (olr or otb):
                self.fill_pixel(im, i, j, 0)
            elif br:
                self.fill_pixel(im, i, j, fill_type)

    def plus(self, im, x, y, sz, fill_type=-1):
        bar_sz = sz // 9
        tl = (x - bar_sz, y - bar_sz)
        tr = (x + bar_sz, y - bar_sz)
        bl = (x - bar_sz, y + bar_sz)
        br = (x + bar_sz, y + bar_sz)
        lb = (x - (sz // 2))
        rb = (x + (sz // 2))
        tb = (y - (sz // 2))
        bb = (y + (sz // 2))
        for i, j in self.iterator(im.shape[0]):
            ls  = (i > tl[0]) # Right of left side
            rs  = (i < tr[0]) # Left of right side
            ts  = (j > tl[1]) # Below top side
            bs  = (j < bl[1]) # Above bottom side
            le  = tl[0] == i  # On left interior edge
            re  = tr[0] == i  # On right interior edge
            te  = tl[1] == j  # On top interior edge
            be  = bl[1] == j  # On bottom interior edge
            r   = (x + (sz // 2)) == i # On right exterior edge
            l   = (x - (sz // 2)) == i # On left exterior edge
            b   = (y + (sz // 2)) == j # On bottom exterior edge
            t   = (y - (sz // 2)) == j # On top exterior edge
            tts = (rs and ls and t)
            rrs = (ts and bs and r)
            bbs = (rs and ls and b)
            lls = (ts and bs and l)
            trs = (re and bs and not ts)
            brs = (re and not bs and ts)
            tls = (le and bs and not ts)
            bls = (le and not bs and ts)
            rts = (te and ls and not rs)
            rbs = (be and ls and not rs)
            lts = (te and not ls and rs)
            lbs = (be and not ls and rs)
            e1 = i < lb
            e2 = i > rb
            e3 = j < tb
            e4 = j > bb
            e = e1 or e2 or e3 or e4
            outer_edges = tts or rrs or bbs or lls
            result = outer_edges or trs or brs or tls or bls or rts or rbs or lts or lbs
            result = result and not e
            if result:
                self.fill_pixel(im, i, j, 0)
            r1 = ls and rs and not e
            r2 = ts and bs and not e
            if (r1 or r2) and not outer_edges:
                self.fill_pixel(im, i, j, fill_type)
            

    def pacman(self, im, x, y, sz, fill_type=-1):
        r = sz // 2
        for i, j in self.iterator(im.shape[0]):
            val = (x-i)**2 + (y-j)**2
            if not (i > x and j < y):
                if val < (r**2 + r) and val > (r**2 - r):
                    self.fill_pixel(im, i, j, 0)
                elif val < r**2:
                    self.fill_pixel(im, i, j, fill_type)
                if i == x and j < y and j > (y - r):
                    self.fill_pixel(im, i, j, 0)
                elif i > x and i < (x + r) and j == y:
                    self.fill_pixel(im, i, j, 0)
        self.fill(im, x - 3, y + 3, im.shape[0], fill_type)

    # def pentagon(self, im, x, y, r, fill_type=-1):
    #     pass

    # def hexagon(self, im, x, y, r, fill_type=-1):
    #     pass

    # def slash_box(self, im, x, y, sz, fill_type=-1);
    #     pass

    # def half_box(self, im, x, y, sz, fill_type=-1):
    #     pass

    # def half_circle(self, im, x, y, r, fill_type=-1):
    #     pass

    def neighbors(self):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i==j or i==-j):
                    yield i, j

    def fill(self, im, x, y, sz, fill_type=1):
        start_node = (x, y)
        expanded = []
        unexpanded = [start_node]
        while unexpanded:
            node = unexpanded.pop()
            expanded.append(node)
            self.fill_pixel(im, node[0], node[1], fill_type)
            for i, j in self.neighbors():
                y1 = node[1] + j
                x1 = node[0] + i
                if not(y1 >= sz or y1 < 0 or x1 >= sz or x1 < 0):
                    if not (node[0]+i, node[1]+j) in expanded and not (node[0]+i, node[1]+j) in unexpanded:
                        if not im[node[1]+j][node[0]+i] == 0:
                            unexpanded.append((node[0]+i, node[1]+j))

    def heart(self, im, x, y, r, fill_type=-1):
        boundary = dict()
        avg_y = []
        for t in range(r * 24):
            i = 16 * math.sin(t)**3
            j = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
            i = (x + i * r / (10 * math.pi))
            j = y - (j * r / (10 * math.pi)) - (r//7)
            if not int(j) in boundary.keys():
                boundary[int(j)] = [int(i)]
            else:
                boundary[int(j)].append(int(i))
            self.fill_pixel(im, int(i), int(j), 0)
        
        self.fill(im, x, y, im.shape[0], fill_type)

    def compound_shape(self, im, x, y, r, fill_type, f1, f2):
        f1(im, x, y, r, -1)
        f2(im, x, y, r//3, fill_type)

    def multi_shape(self, shape, r, N, fill_type=-1):
        im = self.gen(shape, r, 49, -1)
        c = im.shape[0]//2
        for n in range(N-1):
            shape(im, c, c, r - (r//N)*(n+1), -1)
        return im

    def multi_shapes(self, shapes, r, fill_type=-1):
        im = self.gen(shapes[0], r, 49, -1)
        c = im.shape[0]//2
        for n in range(1, len(shapes)):
            shapes[n](im, c, c, r - (r//len(shapes))*(n), -1)
        return im

    def multi_fill_shape(self, shape, r, N, fill_type=-1):
        im = self.gen(shape, r, 199, fill_type)
        c = im.shape[0]//2
        for n in range(1, 2*N):
            r1 = r - int((r/N)*n/2)
            f1 = -1
            f2 = fill_type
            even = n % 2 == 0
            f3 = f2 if even else f1
            shape(im, c, c, r1, f3)
        return im

    def multi_fill_shapes(self, shapes, r, fill_type=-1):
        N = len(shapes)
        im = self.gen(shapes[0], r, 199, fill_type)
        c = im.shape[0]//2
        for n in range(1, 2*N):
            r1 = r - int((r/N)*n/2)
            f1 = -1
            f2 = fill_type
            even = (n//2) % 2 == 0
            shape = shapes[n//2]
            f3 = f2 if (n % 2) == 0 else f1
            shape(im, c, c, r1, f3)
        return im

    def gen(self, f, r, sz=49, fill_type=-1):
        im = self.gen_blank_image(sz)
        f(im, sz//2, sz//2, r, fill_type)
        return im

    def corners(self, f, fill_type=-1):
        c = [(12,12),(12,36),(36,12),(36,36)]
        imgs = []
        for p in c:
            im = self.gen_blank_image()
            f(im, p[0], p[1], 20, fill_type)
            imgs.append(im)
        return imgs

    def corners_double(self, f, f2, fill_type=-1):
        c = [(12,12),(12,36),(36,12),(36,36)]
        imgs = []
        for i, p in enumerate(c):
            im = self.gen_blank_image()
            f(im, p[0], p[1], 20, -1)
            f2(im, p[0], p[1], 8, fill_type)
            imgs.append(im)
        return imgs

    def rotate(self, np_im, rotation):
        tmp = Image.fromarray(self.gen_blank_image(99))
        im = Image.fromarray(np_im).convert('L')
        tmp.paste(im, (25, 25))
        rot = tmp.rotate(rotation, expand=False)
        # x = int((rotation % 180) / 100) * 24
        rot.save("rot_test.png")
        region = rot.crop((25, 25, 74, 74))
        return np.array(region)

    def reflect(self, np_im, ref):
        tmp = Image.fromarray(np_im).convert('L')
        reflection = tmp.transpose(ref)
        return np.array(reflection)

    def save(self, im, name, shrink=True):
        im = Image.fromarray(im).convert('L')
        if shrink:
            im = self.image_processor.shrink_image(im, 24)
        im.save(name)

    def gen_reflections(self):
        ref_dir = Path.cwd() / 'reflections'
        if not ref_dir.exists():
            Path.mkdir(ref_dir)
        refs = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        for i, shape in enumerate(self.all_shapes):
            for fill_type in range(-1, 2):
                corners = self.corners(shape, fill_type)
                for j, corner in enumerate(corners):
                    name = 'reflections/0_0_1_' + str(fill_type+1) + '_0_' + str(i) + '_' + str(j) + '.png'
                    self.save(corner, name)
                    for k, ref in enumerate(refs):
                        name = 'reflections/0_' + str(k+1) + '_1_' + str(fill_type+1) + '_0_' + str(i) + '_' + str(j) + '.png'
                        reflection = self.reflect(corner, ref)
                        self.save(reflection, name)
        for i, shape in enumerate(self.all_shapes):
            for l, shape2 in enumerate(self.all_shapes):
                for fill_type in range(-1, 2):
                    corners = self.corners_double(shape, shape2, fill_type)
                    for j, corner in enumerate(corners):
                        name = 'reflections/0_0_1_' + str(0) + '_0_' + str(i) + '_' + str(j) + '_' + str(l) +'.png'
                        self.save(corner, name)
                        for k, ref in enumerate(refs):
                            name = 'reflections/0_' + str(k+1) + '_1_' + str(0) + '_0_' + str(i) + '_' + str(j) + '_' + str(l) + '.png'
                            reflection = self.reflect(corner, ref)
                            self.save(reflection, name)
        for i, shape in enumerate(self.reflection_shapes):
            for l, shape2 in enumerate(self.all_shapes):
                for fill_type in range(-1, 2):
                    for rot_shape, rotation in self.gen_rotation_shapes_double(shape, shape2, fill_type, 360):
                        name = 'reflections/' + str(rotation) + '_' + str(0) + '_2_' + str(0) + '_0_' + str((i+1)*10) + '_' + str(100*(j+1)) + '_' + str((l+1)*1000) + '.png'
                        self.save(rot_shape, name)
                        for k, ref in enumerate(refs):
                            name = 'reflections/' + str(rotation) + '_' + str(k+1) + '_2_' + str(0) + '_0_' + str((i+1)*10) + '_' + str(100*(j+1)) + '_' + str((l+1)*1000) +'.png'
                            reflection = self.reflect(rot_shape, ref)
                            self.save(reflection, name)
        for i, shape in enumerate(self.reflection_shapes):
            for fill_type in range(-1, 2):
                for rot_shape, rotation in self.gen_rotation_shapes(shape, fill_type, 360):
                    name = 'reflections/' + str(rotation) + '_' + str(0) + '_2_' + str(fill_type+1) + '_0_' + str((i+1)*1000) + '_' + str(j) + '.png'
                    self.save(rot_shape, name)
                    for k, ref in enumerate(refs):
                        name = 'reflections/' + str(rotation) + '_' + str(k+1) + '_2_' + str(fill_type+1) + '_0_' + str((i+1)*1000) + '_' + str(j) +'.png'
                        reflection = self.reflect(rot_shape, ref)
                        self.save(reflection, name)


    def gen_rotation_shapes(self, shape1, fill_type, limit):
        for rotation in range(0, 360, 15):
            im = self.gen(shape1, 36, 49, fill_type)
            im = self.rotate(im, rotation)
            yield im, rotation

    def gen_rotation_shapes_double(self, shape1, shape2, fill_type, limit):
        for rotation in range(0, 360, 15):
            im = self.gen(shape1, 36, 49, -1)
            shape2(im, 24, 24, 12, fill_type)
            im = self.rotate(im, rotation)
            yield im, rotation

    def gen_rotations(self):
        rot_dir = Path.cwd() / 'rotations'
        if not rot_dir.exists():
            Path.mkdir(rot_dir)
        for i, shape in enumerate(self.full_rotation_shapes):
            for fill_type in range(-1, 2):
                for rotation in range(0, 360, 15):
                    name = str(rotation) + '_0_1_' + str(fill_type+1) + '_0_' + str(i) + '.png'
                    im = self.gen(shape, 36, 49, fill_type)
                    im = self.rotate(im, rotation)
                    self.save(im, 'rotations/' + name)
        for i, shape in enumerate(self.full_rotation_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                for fill_type in range(-1, 2):
                    for rotation in range(0, 360, 15):
                        name = str(rotation) + '_0_1_' + str(0) + '_0_' + str(i) + '_' + str(j) + '.png'
                        im = self.gen_blank_image(49)
                        shape(im, 24, 24, 36, -1)
                        shape2(im, 24, 24, 12, fill_type)
                        im = self.rotate(im, rotation)
                        self.save(im, 'rotations/' + name)
        for i, shape in enumerate(self.partial_rotation_shapes):
            for fill_type in range(-1, 2):
                for rotation in range(0, 90, 15):
                    name = str(rotation) + '_0_1_' + str(fill_type+1) + '_0_' + str((i+1)*10) + '.png'
                    im = self.gen(shape, 36, 49, fill_type)
                    im = self.rotate(im, rotation)
                    self.save(im, 'rotations/' + name)
        for i, shape in enumerate(self.partial_rotation_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                for fill_type in range(-1, 2):
                    for rotation in range(0, 90, 15):
                        name = str(rotation) + '_0_1_' + str(0) + '_0_' + str((i+1)*10) + '_' + str(j) + '.png'
                        im = self.gen_blank_image(49)
                        shape(im, 24, 24, 36, -1)
                        shape2(im, 24, 24, 12, fill_type)
                        im = self.rotate(im, rotation)
                        self.save(im, 'rotations/' + name)
    

    def gen_numbers(self):
        num_dir = Path.cwd() / 'numbers'
        if not num_dir.exists():
            Path.mkdir(num_dir)
        for i, shape in enumerate(self.all_shapes):
            for j in range(1, 7):
                name = 'numbers/0_0_' + str(j) + '_0_0_' + str(i) + '_-1_' + str(j) + '.png'
                im = self.multi_shape(shape, 40, j, -1)
                self.save(im, name, True)
        for i, shape1 in enumerate(self.all_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                shapes = []
                for k in range(1, 7):
                    to_append = shape1 if (k % 2 == 0) else shape2
                    shapes.append(to_append)
                name = 'numbers/0_0_' + str(len(shapes)) + '_0_0_' + str(i) + '_' + str(j) + '_' + str(len(shapes)) + '.png'
                im = self.multi_shapes(shapes, 40, -1)
                self.save(im, name, True)
                shapes = []
        for i, shape in enumerate(self.all_shapes):
            for fill_type in range(0, 2):
                if (shape1 == self.heart) and fill_type == 0:
                        continue
                for j in range(1, 7):
                    name = 'numbers/0_0_' + str(j) + '_' + str(fill_type+1) + '_0_0_' + str(i) + '_-1_' + str(j) + '.png'
                    im = self.multi_fill_shape(shape, 160, j, fill_type)
                    self.save(im, name, True)
        for i, shape1 in enumerate(self.all_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                shapes = []
                for fill_type in range(0, 2):
                    if (shape1 == self.heart or shape2 == self.heart) and fill_type == 0:
                        continue
                    for k in range(2, 7):
                        to_append = shape1 if (k % 2 == 0) else shape2
                        shapes.append(to_append)
                    name = 'numbers/0_0_' + str(len(shapes)) + '_' + str(fill_type+1) + '_0_' + str(i) + '_' + str(j) + '_' + str(len(shapes)) + '.png'
                    im = self.multi_fill_shapes(shapes, 160, fill_type)
                    self.save(im, name, True)
                    shapes = []

    def gen_shapes(self):
        for j in range(10):
            for i, shape in enumerate(self.all_shapes):
                if shape == self.pacman:
                    for k in range(4):
                        if shape == self.pacman:
                            label = [1, 0]
                        else:
                            label = [0, 1]
                        for rotation in range(0, 360, 90):
                            im = self.gen_blank_image()
                            shape(im, 24, 24, 30, 1)
                            name = str(label[0]) + '_' + str(label[1]) + '_' + str(rotation) + '_' + str(i + 100 * j + 10000*k) + '.png'
                            im = self.rotate(im, rotation)
                            self.save(im, 'shapes/' + name)
                else:
                    if shape == self.pacman:
                        label = [1, 0]
                    else:
                        label = [0, 1]
                    for rotation in range(0, 360, 90):
                        im = self.gen_blank_image()
                        shape(im, 24, 24, 30, 1)
                        name = str(label[0]) + '_' + str(label[1]) + '_' + str(rotation) + '_' + str(i + 100 * j) + '.png'
                        im = self.rotate(im, rotation)
                        self.save(im, 'shapes/' + name)


    def test(self):
        for i, f in enumerate(self.all_shapes):
            shape_im = self.gen_blank_image()
            f(shape_im, 24, 24, 30, 1)
            im = Image.fromarray(shape_im).convert('L')
            im.save('test' + str(i) + '.png')
        for i, f in enumerate(self.all_shapes):
            for j, f2 in enumerate(self.all_shapes):
                shape_im = self.gen_blank_image()
                self.compound_shape(shape_im, 24, 24, 30, 1, f, f2)
                im = Image.fromarray(shape_im).convert('L')
                im.save('test' + str(i) + '_' + str(j) + '.png')

class Generator2:
    def __init__(self):
        self.g = DataGenerator()
        self.rot_dict =        {}
        self.ref_dict =        {}
        self.idem_dict =       {}
        self.create_dict =     {}
        self.destroy_dict  =   {}
        self.shapeshift_dict = {}
        self.shape_dict      = {}
        self.rot_cmps        = []
        self.ref_cmps        = []
        self.idem_cmps       = []
        self.create_cmps     = []
        self.destroy_cmps    = []
        self.dicts           = {}
        self.all_keys        = {}
        self.shape_list      = []
        self.rot_list        = []
        self.ref_list        = []
        self.idem_list       = []
        self.destroy_list    = []
        self.create_list     = []
        self.init_dicts()
        self.gen_shapes()

    def save(self, im, name, shrink=True):
        im = Image.fromarray(im).convert('L')
        # if shrink:
        #     im = self.image_processor.shrink_image(im, 24)
        im.save(name)

    def gen_shapes(self):
        # self.gen_single()
        # print("Singles generated...")
        # self.gen_double()
        # print("Doubles generated...")
        # self.gen_triple()
        # print("Triples generated...")
        self.gen_transforms()
        #self.gen_comparisons()

    def init_dicts(self):
        self.dicts[0] = self.rot_dict
        self.dicts[1] = self.ref_dict
        self.dicts[2] = self.idem_dict
        self.dicts[3] = self.create_dict
        self.dicts[4] = self.destroy_dict

    def iter_shapes_double(self):
        for shape1 in self.g.all_shapes:
            for shape2 in self.g.all_shapes:
                yield shape1, shape2

    def get_ims(self):
        im1 = self.g.gen_blank_image()
        im2 = self.g.gen_blank_image()
        return im1, im2

    def glue(self, im1, im2):
        return np.concatenate((im1, im2), axis=0)

    def name_shape(self, shape):
        if shape == self.g.heart:
            return "heart"
        elif shape == self.g.square:
            return "square"
        elif shape == self.g.pacman:
            return "pacman"
        elif shape == self.g.circle:
            return "circle"
        elif shape == self.g.triangle:
            return "triangle"
        elif shape == self.g.right_triangle:
            return "righttriangle"
        elif shape == self.g.plus:
            return "plus"
        return str(-2)

    def name_image(self, shape1, shape2, shape3, f1, f2, f3, transform, transform_arg):
        s1 = self.name_shape(shape1)
        s2 = self.name_shape(shape2)
        s3 = self.name_shape(shape3)
        f1 = str(f1)
        f2 = str(f2)
        f3 = str(f3)
        t  = str(transform)
        a  = str(transform_arg)
        n  = '_'.join([s1, s2, s3, f1, f2, f3, t])
        n += '_' + str(a) + '.png'
        return n

    def name_image_pair(self, traits, transform, transform_arg):
        s1 = traits[0]
        s2 = traits[1]
        s3 = traits[2]
        f1 = traits[3]
        f2 = traits[4]
        f3 = traits[5]
        t  = str(transform)
        a  = str(transform_arg)
        n  = '_'.join([s1, s2, s3, f1, f2, f3, t])
        n += '_' + str(a) + '.png'
        return n

    def gen_single(self):
        for shape in self.g.all_shapes:
            for f in range(-1, 2):
                im = self.g.gen_blank_image()
                shape(im, 24, 24, 30, f)
                name = self.name_image(shape, -2, -2, f, -2, -2, "none", 0)
                self.save(im, 'basic_shapes/' + name)
                self.shape_dict[name] = im
    
    def gen_double(self):
        for s1, f1, s2, f2 in self.iter_double():
            im = self.g.gen_blank_image()
            s1(im, 24, 24, 30, f1)
            s2(im, 24, 24, 20, f2)
            name = self.name_image(s1, s2, -2, f1, f2, -2, "none", 0)
            self.save(im, 'basic_shapes/' + name)
            self.shape_dict[name] = im

    def gen_triple(self):
        for s1, f1, s2, f2, s3, f3 in self.iter_triple():
            im = self.g.gen_blank_image()
            s1(im, 24, 24, 30, f1)
            s2(im, 24, 24, 20, f2)
            s3(im, 24, 24, 10, f3)
            name = self.name_image(s1, s2, s3, f1, f2, f3, "none", 0)
            self.save(im, 'basic_shapes/' + name)
            self.shape_dict[name] = im

    def iter_double(self):
        for s1 in self.g.all_shapes:
            for f1 in range(-1, 2):
                for s2 in self.g.all_shapes:
                    for f2 in range(-1, 2):
                        if (f1 == 0 and f2 == 0):
                            continue
                        if (s2 == self.g.heart and f1 == 0):
                            continue
                        yield s1, f1, s2, f2

    def iter_triple(self):
        for s1 in self.g.all_shapes:
            for f1 in range(-1, 2):
                for s2 in self.g.all_shapes:
                    for f2 in range(-1, 2):
                        for s3 in self.g.all_shapes:
                            for f3 in range(-1, 2):
                                if(f1 == 0 and f2 == 0 and f3 == 0) or (f1 == 0 and f2 == 0) or (f2 == 0 and f3== 0):
                                    continue
                                if(s3 == self.g.heart and f2 == 0):
                                    continue
                                if(s2 == self.g.heart and f1 == 0):
                                    continue
                                yield s1, f1, s2, f2, s3, f3

    def parse_name(self, name):
        n = name.split('_')
        s1 = n[0]
        s2 = n[1]
        s3 = n[2]
        f1 = n[3]
        f2 = n[4]
        f3 = n[5]
        t  = n[6]
        a  = n[7]
        return (s1, s2, s3, f1, f2, f3, t, a)

    def is_rot(self, traits):
        match = ['righttriangle, pacman, triangle, heart']
        return traits[0] in match or traits[1] in match or traits[2] in match

    def is_part_rot(self, traits):
        match = ['square', 'plus']
        return traits[0] in match or traits[1] in match or traits[2] in match

    def gen_rots(self):
        for key in self.shape_dict.keys():
            shape   = self.shape_dict[key]
            traits  = self.parse_name(key)
            if self.is_rot(traits):
                for rotation in range(0, 360, 45):
                    im1  = shape
                    im2  = self.g.rotate(im1, rotation)
                    im   = self.glue(im1, im2)
                    name = self.name_image_pair(traits, "rotation", str(rotation))
                    self.save(im, 'rot_pairs/' + name)
                    self.rot_dict[name] = im
            if self.is_part_rot(traits):
                for rotation in range(0, 90, 45):
                    im1  = shape
                    im2  = self.g.rotate(im1, rotation)
                    im   = self.glue(im1, im2)
                    name = self.name_image_pair(traits, "rotation", str(rotation))
                    self.save(im, 'rot_pairs/' + name)
                    self.rot_dict[name] = im

    def corners(self, s, f):
        c = [(12,12),(12,36),(36,12),(36,36)]
        for p in c:
            im = self.g.gen_blank_image()
            s(im, p[0], p[1], 20, f)
            print(s)
            name = self.name_image(s, -2, -2, f, -2, -2, "reflection", -1)
            print(name)
            self.ref_shapes_dict[name] = im

    def corners_double(self, s1, s2, f1, f2):
        c = [(12,12),(12,36),(36,12),(36,36)]
        for i, p in enumerate(c):
            im = self.g.gen_blank_image()
            s1(im, p[0], p[1], 20, f1)
            s2(im, p[0], p[1], 10, f2)
            name = self.name_image(s1, s2, -2, f1, f2, -2, "reflection", -1)
            self.ref_shapes_dict[name] = im
            #self.save(im, name)

    def corners_triple(self, s1, s2, s3, f1, f2, f3):
        c = [(12,12),(12,36),(36,12),(36,36)]
        for i, p in enumerate(c):
            im = self.g.gen_blank_image()
            s1(im, p[0], p[1], 20, f1)
            s2(im, p[0], p[1], 10, f2)
            s3(im, p[0], p[1], 5, f3)
            name = self.name_image(s1, s2, s3, f1, f2, f3, "reflection", -1)
            self.ref_shapes_dict[name] = im
            #self.save(im, name)

    def is_ref(self, t):
        match = ['righttriangle', 'pacman']
        return t[0] in match or t[1] in match or t[2] in match

    def gen_ref_shapes(self):
        self.ref_shapes_dict = {}
        for s in self.g.all_shapes:
            for f in range(-1, 2):
                self.corners(s, f)
        for s1, f1, s2, f2 in self.iter_double():
            self.corners_double(s1, s2, f1, f2)
        # for s1, f1, s2, f2, s3, f3 in self.iter_triple():
        #     self.corners_triple(s1, s2, s3, f1, f2, f3)

    def gen_refs(self):
        self.gen_ref_shapes()
        for key in self.ref_shapes_dict.keys():
            s = self.ref_shapes_dict[key]
            traits = self.parse_name(key)
            refs   = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
            for i, ref in enumerate(refs):
                im1  = s
                im2  = self.g.reflect(im1, ref)
                im   = self.glue(im1, im2)
                name = self.name_image_pair(traits, "reflection", str(i+1))
                print(traits)
                self.save(im, 'ref_pairs/' + name)
                self.ref_list.append(name)
        for key in self.shape_dict.keys():
            shape   = self.shape_dict[key]
            traits  = self.parse_name(key)
            if not self.is_ref(traits):
                continue
            refs    = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
            for i, ref in enumerate(refs):
                im1  = shape
                im2  = self.g.reflect(im1, ref)
                im   = self.glue(im1, im2)
                name = self.name_image_pair(traits, "reflection", str(i+1))
                self.save(im, 'ref_pairs/' + name)
                self.ref_list.append(name)

    def gen_create(self):
        for key in self.shape_dict.keys():
            shape   = self.shape_dict[key]
            traits  = self.parse_name(key)
            for key2 in self.shape_dict.keys():
                shape2 = self.shape_dict[key2]
                traits2 = self.parse_name(key2)
                match, s = self.find_create_pair(traits, traits2)
                if match:
                    im1  = shape
                    im2  = shape2
                    im   = self.glue(im1, im2)
                    name1 = self.name_image_pair(traits, "create", str(s))
                    name2 = self.name_image_pair(traits2, "create", str(s))
                    name = name1.split('.')[0] + "sep" + name2
                    self.save(im, 'create_pairs/' + name)
                    self.create_list.append(name)

    def find_create_pair(self, trait1, trait2):
        t1s1 = trait1[0]
        t1s2 = trait1[1]
        t1s3 = trait1[2]
        t2s1 = trait2[0]
        t2s2 = trait2[1]
        t2s3 = trait2[2]
        f1   = trait1[3] == trait2[3]
        f2   = trait1[4] == trait2[4]
        s1   = t1s1 == t2s1
        s2   = t1s2 == t2s2
        s3   = t1s3 == t2s3
        m1   = s1 and f1 and t1s2 == str(-2) and t2s2 != str(-2)
        m2   = s1 and s2 and f1 and f2 and t1s3 == str(-2) and t2s3 != str(-2)
        if m1:
            return True, trait2[1]
        if m2:
            return True, trait2[2]
        return False, 0

    def gen_destroy(self):
        for key in self.shape_dict.keys():
            shape   = self.shape_dict[key]
            traits  = self.parse_name(key)
            for key2 in self.shape_dict.keys():
                shape2 = self.shape_dict[key2]
                traits2 = self.parse_name(key2)
                match, s = self.find_destroy_pair(traits, traits2)
                if match:
                    im1  = shape
                    im2  = shape2
                    im   = self.glue(im1, im2)
                    name1 = self.name_image_pair(traits, "destroy", str(s))
                    name2 = self.name_image_pair(traits2, "destroy", str(s))
                    name  = name1.split('.')[0] + "sep" + name2
                    self.save(im, 'destroy_pairs/' + name)
                    self.destroy_list.append(name)

    def find_destroy_pair(self, trait1, trait2):
        t1s1 = trait1[0]
        t1s2 = trait1[1]
        t1s3 = trait1[2]
        t2s1 = trait2[0]
        t2s2 = trait2[1]
        t2s3 = trait2[2]
        f1   = trait1[3] == trait2[3]
        f2   = trait1[4] == trait2[4]
        s1   = t1s1 == t2s1
        s2   = t1s2 == t2s2
        s3   = t1s3 == t2s3
        m1   = s1 and f1 and t1s2 != str(-2) and t2s2 == str(-2)
        m2   = s1 and s2 and f1 and f2 and t1s3 != str(-2) and t2s3 == str(-2)
        if m1:
            return True, trait2[1]
        if m2:
            return True, trait2[2]
        return False, 0
        

    def gen_idem(self):
        for key in self.shape_dict.keys():
            shape   = self.shape_dict[key]
            traits  = self.parse_name(key)
            im1  = shape
            im2  = np.copy(im1)
            im   = self.glue(im1, im2)
            name = self.name_image_pair(traits, "identity", str(0))
            self.save(im, 'idem_pairs/' + name)
            self.idem_list.append(name)

    def cmp_rots(self, r1, r2):
        return r1[-1] == r2[-1]

    

    # def gen_rot_cmps(self):
    #     count = 0
    #     for k1 in self.all_rots:
    #         for k2 in self.all_rots:
    #             rot1 = np.array(Image.open(k1[1]))
    #             rot2 = np.array(Image.open(k2[1]))
    #             r1   = self.parse_name(k1)
    #             r2   = self.parse_name(k2)
    #             if self.cmp_rots(r1, r2):
    #                 im = np.concatenate((rot1, rot2), axis=1)
    #                 name = "1_" + self.name_image_pair(r1, "rotation", r1[-1])
    #                 self.save(im, "comparisons/" + name, False)
    #                 count += 1
    #     self.shuffle_keys()
    #     for k1 in range(count):
    #             rot = np.array(Image.open(self.all_rots[k1][1]))
    #             k2 = 0
    #             ran = 0
    #             t1  = self.parse_name(k1)
    #             for k, r in self.iter_rand():
    #                 k2 = k
    #                 ran = r
    #                 t2  = self.parse_name(k2)
    #                 if not self.cmp_rots(t1, t2) or t1[-2] != t2[-2]:
    #                     break
    #             im = np.concatenate((rot, ran), axis=1)
    #             name = "0_" + self.name_image_pair(t1, "invalid", 0)
    #             self.save(im, "comparisons/"+name, False)

    # def gen_ref_cmps(self):
    #     for k1 in self.ref_dict.keys():
    #         for k2 in self.ref_dict.keys():
    #             ref1 = self.ref_dict[k1]
    #             ref2 = self.ref_dict[k2]
    #             r1   = self.parse_name(k1)
    #             r2   = self.parse_name(k2)
    #             if self.cmp_rots(r1, r2):
    #                 im = np.concatenate((ref1, ref2), axis=1)
    #                 name = "1_" + self.name_image_pair(r1, "reflection", r1[-1])
    #                 self.save(im, "comparisons/" + name, False)
    #     self.shuffle_keys()
    #     for k1 in self.ref_dict.keys():
    #         for k2 in self.ref_dict.keys():
    #             rot = self.ref_dict[k1]
    #             k2 = 0
    #             ran = 0
    #             t1  = self.parse_name(k1)
    #             for k, r in self.iter_rand():
    #                 k2 = k
    #                 ran = r
    #                 t2  = self.parse_name(k2)
    #                 if not self.cmp_rots(t1, t2) or t1[-2] != t2[-2]:
    #                     break
    #             im = np.concatenate((rot, ran), axis=1)
    #             name = "0_" + self.name_image_pair(t1, "invalid", 0)
    #             self.save(im, "comparisons/"+name, False)

    def gen_cmp(self, tlist, tname, nlist, cmp_f):
        used = []
        tlist2 = [x for x in tlist]
        for k1 in tlist:
            random.shuffle(tlist2)
            for k2 in tlist2:
                if k1 in used:
                    break
                ref1 = np.array(Image.open(k1[0]))
                ref2 = np.array(Image.open(k2[0]))
                r1   = self.parse_name(k1[1])
                r2   = self.parse_name(k2[1])
                if tname == "rotation":
                    print(r1)
                    print(r2)
                if cmp_f(r1, r2):
                    im = np.concatenate((ref1, ref2), axis=1)
                    name = "1_" + self.name_image_pair(r1, tname, r1[-1])
                    name = name.split('.')[0] + self.name_image_pair(r2, tname, r2[-1])
                    self.save(im, "comparisons/" + name, False)
                    used.append(k1)
        for i, k1 in enumerate(tlist):
            rot = np.array(Image.open(k1[0]))
            k2 = np.array(Image.open(nlist[i][0]))            
            im = np.concatenate((rot, k2), axis=1)
            t1 = self.parse_name(k1[1])
            name = "0_" + self.name_image_pair(t1, "invalid", 0)
            self.save(im, "comparisons/"+name, False)

    def create_cmp(self, trait1, trait2):
        t1s1 = trait1[0]
        t1s2 = trait1[1]
        t1s3 = trait1[2]
        t2s1 = trait2[0]
        t2s2 = trait2[1]
        t2s3 = trait2[2]
        f1   = trait1[3] == trait2[3]
        f2   = trait1[4] == trait2[4]
        s1   = t1s1 == t2s1
        s2   = t1s2 == t2s2
        s3   = t1s3 == t2s3
        m1   = s1 and f1 and t1s2 == str(-2) and t2s2 != str(-2)
        m2   = s1 and s2 and f1 and f2 and t1s3 == str(-2) and t2s3 != str(-2)
        if m2:
            return True, trait2[2]
        if m1:
            return True, trait2[1]
        return False, 0

    def destroy_cmp(self, trait1, trait2):
        t1s1 = trait1[0]
        t1s2 = trait1[1]
        t1s3 = trait1[2]
        t2s1 = trait2[0]
        t2s2 = trait2[1]
        t2s3 = trait2[2]
        f1   = trait1[3] == trait2[3]
        f2   = trait1[4] == trait2[4]
        s1   = t1s1 == t2s1
        s2   = t1s2 == t2s2
        s3   = t1s3 == t2s3
        m1   = s1 and f1 and t1s2 != str(-2) and t2s2 == str(-2)
        m2   = s1 and s2 and f1 and f2 and t1s3 != str(-2) and t2s3 == str(-2)
        if m2:
            return True, trait2[2]
        if m1:
            return True, trait2[1]
        return False, 0

    def gen_rot_cmps(self):
        self.gen_cmp(self.all_rots, "rotation", self.all_but_rots, self.cmp_rots)

    def gen_ref_cmps(self):
        self.gen_cmp(self.all_refs, "reflection", self.all_but_refs, self.cmp_rots)

    def gen_idem_cmps(self):
        self.gen_cmp(self.all_idem, "identity", self.all_but_idem, lambda a, b: a == b)

    def gen_destroy_cmps(self):
        self.gen_cmp(self.all_dest, "destroy", self.all_but_dest, self.destroy_cmp)

    def gen_create_cmps(self):
        self.gen_cmp(self.all_crea, "create", self.all_but_crea, self.create_cmp)
    ################################## Possible error
    
    
    def iter_rand(self):
        for idx in range(self.min_idx()):
            for k in self.dicts.keys():
                yield k, self.all_keys[k][idx]

    def min_idx(self):
        m = 99999
        for d in self.dicts:
            if len(d) > m:
                m = len(d)
        return m

    def shuffle_keys(self):
        for key in self.dicts.keys():
            d = self.dicts[key]
            k = list(d.keys())
            np.random.shuffle(k)
            self.all_keys[key] = k

    def gen_comparisons(self):
        self.gen_rot_cmps()
        print("Rotations complete...")
        self.gen_ref_cmps()
        print("Reflections complete...")
        self.gen_idem_cmps()
        print("Identities complete...")
        self.gen_create_cmps()
        print("Creates complete...")
        self.gen_destroy_cmps()
        print("Destroys complete...")

    def load_lists(self):
        self.all_rots = []
        self.all_refs = []
        self.all_idem = []
        self.all_crea = []
        self.all_dest = []
        rot_path = Path.cwd() / 'rot_pairs'
        ref_path = Path.cwd() / 'ref_pairs'
        ide_path = Path.cwd() / 'idem_pairs'
        cre_path = Path.cwd() / 'create_pairs'
        des_path = Path.cwd() / 'destroy_pairs'
        self.load_list(self.all_rots, rot_path)
        self.load_list(self.all_refs, ref_path)
        self.load_list(self.all_idem, ide_path)
        self.load_list(self.all_crea, cre_path)
        self.load_list(self.all_dest, des_path)
        self.all_but_rots = self.all_refs + self.all_idem + self.all_crea + self.all_dest
        self.all_but_refs = self.all_rots + self.all_idem + self.all_crea + self.all_dest
        self.all_but_crea = self.all_refs + self.all_idem + self.all_rots + self.all_dest
        self.all_but_dest = self.all_refs + self.all_idem + self.all_crea + self.all_rots
        self.all_but_idem = self.all_refs + self.all_rots + self.all_crea + self.all_dest
        random.shuffle(self.all_but_rots)
        random.shuffle(self.all_but_refs)
        random.shuffle(self.all_but_crea)
        random.shuffle(self.all_but_dest)
        random.shuffle(self.all_but_idem)

    def load_list(self, l, p):
        for f in p.iterdir():
            if '.png' in f.name:
                l.append((f, f.name))

    def gen_transforms(self):
        #self.gen_rots()
        print("Rotations generated...")
        #self.gen_refs()
        print("Reflections generated...")
        #self.gen_idem()
        print("Identities generated...")
        # self.gen_create()
        # print("Creates generated...")
        # self.gen_destroy()
        #print("Destroys generated...")
        # self.gen_shapeshift()
        self.load_lists()
        self.gen_comparisons()
        print("Comparisons generated...")


if __name__ == "__main__":
    g = Generator2()
    #g.gen_rotations()
    #g.gen_numbers()
    #g.gen_reflections()
