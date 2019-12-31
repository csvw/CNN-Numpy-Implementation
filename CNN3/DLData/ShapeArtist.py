import numpy as np
from PIL import Image
from pathlib import Path
import math
from ImagePreprocessor import ImagePreprocessor
import random

class ShapeArtist:
    def __init__(self):
        self.image_processor = ImagePreprocessor()
        self.partial_rotation_shapes = [self.square, self.diamond, self.plus]
        self.all_shapes = []
        self.full_rotation_shapes = [self.pacman, self.triangle, self.heart, self.right_triangle]
        self.reflection_shapes = [self.pacman, self.right_triangle]
        self.shape_names = {}
        self.name_to_shape = {}
        self.shape_numbers = {}
        self.init_all_shapes()

    def init_all_shapes(self):
        self.all_shapes.append(self.circle)
        self.all_shapes.append(self.square)
        self.all_shapes.append(self.heart)
        self.all_shapes.append(self.pacman)
        self.all_shapes.append(self.triangle)
        self.all_shapes.append(self.right_triangle)
        self.all_shapes.append(self.plus)
        self.all_shapes.append(self.star)
        self.all_shapes.append(self.hexagon)
        self.all_shapes.append(self.octagon)
        self.shape_names[self.circle] = "Circle"
        self.shape_names[self.square] = "Square"
        self.shape_names[self.heart] = "Heart"
        self.shape_names[self.pacman] = "Pacman"
        self.shape_names[self.triangle] = "Triangle"
        self.shape_names[self.right_triangle] = "RightTriangle"
        self.shape_names[self.plus] = "Plus"
        self.shape_names[self.star] = "Star"
        self.shape_names[self.hexagon] = "Hexagon"
        self.shape_names[self.octagon] = "Octagon"
        self.name_to_shape["Circle"] = self.circle
        self.name_to_shape["Square"] = self.square
        self.name_to_shape["Heart"]  = self.heart
        self.name_to_shape["Pacman"] = self.pacman
        self.name_to_shape["Triangle"] = self.triangle
        self.name_to_shape["RightTriangle"] = self.right_triangle
        self.name_to_shape["Plus"] = self.plus
        self.name_to_shape["Hexagon"] = self.hexagon
        self.name_to_shape["Octagon"] = self.octagon
        for i, s in enumerate(self.all_shapes):
            self.shape_numbers[s] = i

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
        if p2[0] - p1[0] == 0:
            return 999999
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

    def octagon(self, im, x, y, r, fill_type=-1):
        r = r // 2
        slopes = []
        pairs  = []
        c = 0
        pentagon=[]
        R = r
        for n in range(0,8):
            x_ = R*math.cos(math.radians(22.5 + n*45)) + x
            y_ = -R*math.sin(math.radians(22.5 + n*45)) + y
            pentagon.append([x_, y_])
        for i in range(len(pentagon)):
            p1 = pentagon[c]
            p2 = pentagon[(c+1) % 8]
            c += 1
            c %= 8
            if p1[0] - p2[0] == 0:
                m = 999999
            else:
                m  = (p1[1] - p2[1]) / (p1[0] - p2[0])
            slopes.append(m)
            pairs.append((p1, p2))
        for i, j in self.iterator(im.shape[0]):
            draw = True
            for (p1, p2) in pairs:
                t1 = self.left_hand(p1, p2, (i, j))
                if not t1:
                    draw = False
            if draw:
                self.fill_pixel(im, i, j, fill_type)
        for i, j in self.iterator(im.shape[0]):
            for m, (p1, p2) in zip(slopes, pairs):
                y_ = (j - p1[1])
                x_ = m * (i - p1[0])
                s = abs(m)
                s = s * 0.7
                s = 0.8 if s < 0.1 and s > -0.1 else s
                line = x_ >= (y_ - s) and x_ <= (y_ + s) and ((i - x)**2 + (j - y)**2 <= r**2)
                if line:
                    self.fill_pixel(im, i, j, 0)

    def left_hand(self, p1, p2, p):
        x1 = p1[0]
        x2 = p2[0]
        xp = p[0]
        y1 = p1[1]
        y2 = p2[1]
        yp = p[1]
        return ((x2 - x1) * -(yp - y1) - (xp - x1) * -(y2 - y1)) > 0

    def compare_pts(self, p1, p2, p3, p4):
        if (p1[0] + p2[0] + p1[1] + p2[1]) > (p3[0] + p4[0] + p3[1] + p4[1]):
            return 1
        return -1

    def triangle_area(self, p1, p2, p3):
        return abs(0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])))

    def rect_area(self, p1, p2, p3, p4):
        return abs(0.5 * ((p1[0] * p2[1] - p1[1] * p2[0]) + 
                            (p2[0] * p3[1] - p2[1] * p3[0]) +
                            (p3[0] * p4[1] - p3[1] * p4[0]) +
                            (p4[0] * p1[1] - p4[1] * p1[0])))

    def hexagon(self, im, x, y, r, fill_type=-1):
        r = r // 2
        slopes = []
        pairs  = []
        c = 0
        pentagon=[]
        R = r
        for n in range(0,7):
            x_ = R*math.cos(math.radians(n*60)) + x
            y_ = -R*math.sin(math.radians(n*60)) + y
            pentagon.append([x_, y_])
        for i in range(len(pentagon)):
            p1 = pentagon[c]
            p2 = pentagon[(c+1) % 6]
            c += 1
            c %= 6
            if p1[0] - p2[0] == 0:
                m = 0
            else:
                m  = (p1[1] - p2[1]) / (p1[0] - p2[0])
            slopes.append(m)
            pairs.append((p1, p2))
        for i, j in self.iterator(im.shape[0]):
            draw = True
            for (p1, p2) in pairs:
                t1 = self.left_hand(p1, p2, (i, j))
                if not t1:
                    draw = False
            if draw:
                self.fill_pixel(im, i, j, fill_type)
        for i, j in self.iterator(im.shape[0]):
            for m, (p1, p2) in zip(slopes, pairs):
                y_ = (j - p1[1])
                x_ = m * (i - p1[0])
                s = abs(m)
                s = s * 0.7
                s = 0.5 if s < 0.1 and s > -0.1 else s
                line = x_ >= (y_ - s) and x_ <= (y_ + s) and ((i - x)**2 + (j - y)**2 <= r**2)
                if line:
                    self.fill_pixel(im, i, j, 0)
        
    def find_a(self, p1, p2):
        a = .381
        x0 = p2[0] - p1[0]
        y0 = p2[1] - p1[1]
        x  = a * x0
        y  = a * y0
        return (p1[0] + x, p1[1] + y)

    def star(self, im, x, y, sz, fill_type=-1):
        sz = sz // 2
        slopes = []
        pairs  = []
        inR    = 0.35 * sz
        c = 0
        pentagon=[]
        penten=[]
        penten1=[]
        penten2=[]
        penten3=[]
        penten4=[]
        penten5=[]
        R = sz
        a = .381
        b = .236
        a2 = a + b
        for n in range(0,5):
            x_ = R*math.cos(math.radians(90+n*72)) + x
            y_ = -R*math.sin(math.radians(90+n*72)) + y
            pentagon.append([x_, y_])
        for i in range(len(pentagon)):
            p1 = pentagon[c]
            p2 = pentagon[(c+2) % 5]
            p3 = pentagon[(c+1) % 5]
            ap = self.find_a(p1, p2)
            m1 = (p1[1] - ap[1]) / (p1[0] - ap[0])
            m2 = (ap[1] - p3[1]) / (ap[0] - p3[0])
            penten.append((m1, p1, ap))
            penten.append((m2, ap, p3))
            c += 1
            c %= 5
            # print(p1, p2)
            m  = (p1[1] - p2[1]) / (p1[0] - p2[0])
            slopes.append(m)
            pairs.append((p1, p2))

        penten1.append((pentagon[0], pentagon[2]))
        penten1.append((penten[2][1], penten[2][2]))
        penten1.append((penten[3][1], penten[3][2]))
        penten1.append((pentagon[3], pentagon[0]))

        penten2.append((pentagon[1], pentagon[3]))
        penten2.append((penten[8][1], penten[8][2]))
        penten2.append((penten[9][1], penten[9][2]))
        penten2.append((pentagon[4], pentagon[1]))

        penten3.append((pentagon[2], pentagon[4]))
        penten3.append((penten[4][1], penten[4][2]))
        penten3.append((penten[5][1], pentagon[0]))
        penten3.append((pentagon[0], pentagon[2]))

        penten4.append((pentagon[3], pentagon[0]))
        penten4.append((penten[0][1], penten[0][2]))
        penten4.append((penten[1][1], pentagon[1]))
        penten4.append((pentagon[1], pentagon[3]))
        
        penten5.append((pentagon[4], pentagon[1]))
        # penten5.append((pentagon[1], pentagon[2]))
        penten5.append((penten[2][1], penten[2][2]))
        penten5.append((penten[3][1], penten[3][2]))
        penten5.append((pentagon[2], pentagon[4]))


        for i, j in self.iterator(im.shape[0]):
            for m, (p1, p2) in zip(slopes, pairs):
                y_ = (j - p1[1])
                x_ = m * (i - p1[0])
                s = abs(m)
                s = s
                s = 0.5 if s < 0.1 and s > -0.1 else s
                line = x_ >= (y_ - s) and x_ <= (y_ + s) and ((i - x)**2 + (j - y)**2 <= sz**2)
                distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1]) ** 2)
                progress = math.sqrt((p1[0] - i)**2 + (p1[1] - j)**2) / distance
                if line and not (progress > a and progress < a2):
                    # print(p1, p2, progress)
                    self.fill_pixel(im, i, j, 0)
        for i, j in self.iterator(im.shape[0]):
            draw = True
            for p1, p2 in penten1:
                t1 = self.left_hand(p1, p2, (i, j))
                if not t1:
                    draw = False
            if draw:
                self.fill_pixel(im, i, j, fill_type)
            draw = True
            for p1, p2 in penten2:
                t1 = self.left_hand(p1, p2, (i, j))
                if not t1:
                    draw = False
            if draw:
                self.fill_pixel(im, i, j, fill_type)
            draw = True
            for p1, p2 in penten3:
                t1 = self.left_hand(p1, p2, (i, j))
                if not t1:
                    draw = False
            if draw:
                self.fill_pixel(im, i, j, fill_type)
            draw = True
            for p1, p2 in penten5:
                t1 = self.left_hand(p1, p2, (i, j))
                if not t1:
                    draw = False
            if draw:
                self.fill_pixel(im, i, j, fill_type)
        # self.fill(im, x, y, im.shape[0], fill_type)
        # print(penten)
                

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
