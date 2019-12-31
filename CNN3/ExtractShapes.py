from PIL import Image
import numpy as np
from pathlib import Path
from ImagePreprocessor import ImagePreprocessor
from PIL import ImageFilter

class Node:
    def __init__(self, i, j, im):
        self.loc = (i, j)
        self.im = im
        self.friends = []
        self.threshold = 0.05 * 255
        self.v = im[i][j]

    def neighbors(self):
        i, j = self.loc
        candidates = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        neighbors  = []
        for c in candidates:
            if c[0] < 0 or c[1] < 0:
                continue
            if c[0] >= self.im.shape[0] or c[1] >= self.im.shape[1]:
                continue
            neighbors.append(c)
        return neighbors

    def join_hands(self):
        for n in self.neighbors():
            v = self.im[n[0]][n[1]]
            # print(v)
            # print(self.v)
            if v + self.threshold > self.v and v - self.threshold < self.v:
                self.friends.append(n)


class Extractor:
    def __init__(self, im):
        self.p = ImagePreprocessor()
        # self.im = im
        self.im = np.array(im)
        self.nodes = []
        self.node_dict = {}
        self.regions = []
        self.shapes = []
        self.init_nodes()
        self.connect()
        self.collect()
        # self.save()
        
    def init_nodes(self):
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                n = Node(i, j, self.im)
                self.nodes.append(n)
                self.node_dict[(i, j)] = n
        # print("Nodes Initialized...")

    def connect(self):
        for n in self.nodes:
            n.join_hands()
        # print("Nodes Connected...")
        explored = []
        for n in self.nodes:
            region_explored = []
            region_unexplored = []
            if not n in explored:
                region_unexplored.append(n)
                while region_unexplored:
                    cur = region_unexplored.pop()
                    region_explored.append(cur)
                    explored.append(cur)
                    for f in cur.friends:
                        friend = self.node_dict[f]
                        if not friend in region_explored and not friend in region_unexplored:
                            region_unexplored.append(friend)
                            self.nodes.remove(friend)
            if region_explored:
                self.regions.append(region_explored)

    def collect(self):
        for region in self.regions:
            if len(region) < 10:
                    continue
            if region[0].v > 50:
                continue
            self.shapes.append(region)

    def save(self):
        count = 0
        for region in self.regions:
            if len(region) < 10:
                continue
            if region[0].v > 50:
                continue
            im = np.zeros(self.im.shape)
            for i in range(im.shape[0]):
                for j in range(im.shape[0]):
                    if self.node_dict[(i, j)] not in region:
                        im[i][j] = 255
            Image.fromarray(im).convert('L').save(str(count) + ".png")
            count += 1

if __name__ == '__main__':
    path = Path.cwd() / 'a.png'
    im = Image.open(path).convert('L')
    e = Extractor(im)