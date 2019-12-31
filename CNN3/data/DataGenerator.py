import numpy as np
from PIL import Image
from pathlib import Path
import math
from ImagePreprocessor import ImagePreprocessor

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
        for t in range(r * 24):
            i = 16 * math.sin(t)**3
            j = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
            i = (x + i * r / (10 * math.pi))
            j = -(y + j * r / (10 * math.pi)) - (r//7)
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
                        name = 'reflections/' + str(rotation) + '_' + str(0) + '_2_' + str(0) + '_0_' + str(i*10) + '_' + str(j) + '_' + str(l) + '.png'
                        for k, ref in enumerate(refs):
                            name = 'reflections/' + str(rotation) + '_' + str(k+1) + '_2_' + str(0) + '_0_' + str(i*10) + '_' + str(j) + '_' + str(l) +'.png'
                            reflection = self.reflect(rot_shape, ref)
                            self.save(reflection, name)
        for i, shape in enumerate(self.reflection_shapes):
            for fill_type in range(-1, 2):
                for rot_shape, rotation in self.gen_rotation_shapes(shape, fill_type, 360):
                    name = 'reflections/' + str(rotation) + '_' + str(0) + '_2_' + str(fill_type+1) + '_0_' + str(i*10) + '_' + str(j) + '.png'
                    for k, ref in enumerate(refs):
                        name = 'reflections/' + str(rotation) + '_' + str(k+1) + '_2_' + str(fill_type+1) + '_0_' + str(i*10) + '_' + str(j) +'.png'
                        reflection = self.reflect(rot_shape, ref)
                        self.save(reflection, name)


    def gen_rotation_shapes(self, shape1, fill_type, limit):
        for rotation in range(0, 360, 15):
            im = self.gen(shape1, 36, fill_type)
            im = self.rotate(im, rotation)
            yield im, rotation

    def gen_rotation_shapes_double(self, shape1, shape2, fill_type, limit):
        for rotation in range(0, 360, 15):
            im = self.gen(shape1, 36, -1)
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
                    im = self.gen(shape, 36, fill_type)
                    im = self.rotate(im, rotation)
                    self.save(im, 'rotations/' + name)
        for i, shape in enumerate(self.full_rotation_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                for fill_type in range(-1, 2):
                    for rotation in range(0, 360, 15):
                        name = str(rotation) + '_0_1_' + str(0) + '_0_' + str(i) + '_' + str(j) + '.png'
                        im = self.gen_blank_image()
                        shape(im, 24, 24, 36, -1)
                        shape2(im, 24, 24, 12, fill_type)
                        im = self.rotate(im, rotation)
                        self.save(im, 'rotations/' + name)
        for i, shape in enumerate(self.partial_rotation_shapes):
            for fill_type in range(-1, 2):
                for rotation in range(0, 90, 15):
                    name = str(rotation) + '_0_1_' + str(fill_type+1) + '_0_' + str(i*10) + '.png'
                    im = self.gen(shape, 36, fill_type)
                    im = self.rotate(im, rotation)
                    self.save(im, 'rotations/' + name)
        for i, shape in enumerate(self.partial_rotation_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                for fill_type in range(-1, 2):
                    for rotation in range(0, 90, 15):
                        name = str(rotation) + '_0_1_' + str(0) + '_0_' + str(i*10) + '_' + str(j) + '.png'
                        im = self.gen_blank_image()
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
                name = 'numbers/0_0_' + str(j) + '_0_0_' + str(i) + '_' + str(j) + 'N.png'
                im = self.multi_shape(shape, 40, j, -1)
                self.save(im, name, True)
        for i, shape1 in enumerate(self.all_shapes):
            for j, shape2 in enumerate(self.all_shapes):
                shapes = []
                for k in range(1, 7):
                    to_append = shape1 if (k % 2 == 0) else shape2
                    shapes.append(to_append)
                name = 'numbers/0_0_' + str(len(shapes)) + '_0_0_' + str(i) + '_' + str(j) + '_' + str(len(shapes)) + '_' + 'N.png'
                im = self.multi_shapes(shapes, 40, -1)
                self.save(im, name, True)
                shapes = []
        for i, shape in enumerate(self.all_shapes):
            for fill_type in range(0, 2):
                if (shape1 == self.heart) and fill_type == 0:
                        continue
                for j in range(1, 7):
                    name = 'numbers/0_0_' + str(j) + '_' + str(fill_type+1) + '_0_0_' + str(i) + '_' + str(j) + 'N2.png'
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
                    name = 'numbers/0_0_' + str(len(shapes)) + '_' + str(fill_type+1) + '_0_' + str(i) + '_' + str(j) + '_' + str(len(shapes)) + '_' + 'N2.png'
                    im = self.multi_fill_shapes(shapes, 160, fill_type)
                    self.save(im, name, True)
                    shapes = []

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


if __name__ == "__main__":
    g = DataGenerator()
    g.gen_numbers()
