import numpy as np
from PIL import Image
from pathlib import Path

class ShapeLabel:
    def __init__(self, rot, ref, num, fil, changed, id1, id2, id3):
        self.rot = rot
        self.ref = ref
        self.num = num
        self.fil = fil
        self.chg = changed
        self.id1 = id1
        self.id2 = id2
        self.id3 = id3


class PairLabel:
    def __init__(self, rot, ref, num, fil, changed):
        self.rot = rot
        self.ref = ref
        self.num = num
        self.fil = fil
        self.chg = changed


class ShapeData:
    def __init__(self):
        self.cwd      = Path.cwd()
        self.rot_path = self.cwd / 'rotations'
        self.ref_path = self.cwd / 'reflections'
        self.num_path = self.cwd / 'numbers'
        self.rot_imgs = list()
        self.ref_imgs = list()
        self.num_imgs = list()
        self.get_imgs(self.rot_path, self.rot_imgs)
        self.get_imgs(self.ref_path, self.ref_imgs)
        self.get_imgs(self.num_path, self.num_imgs)
        self.rot_dict  = {}
        self.rot_pairs = []
        self.ref_pairs = []
        self.num_pairs = []
        self.fil_pairs = []
        self.chg_pairs = []
        self.init_pairs()
    
    def construct_label(self, name):
        name = name.split('_')
        rot = int(name[0])
        ref = int(name[1])
        num = int(name[2])
        fil = int(name[3])
        chg = int(name[4])
        if len(name) == 6:
            id1 = int(name[5].split('.')[0])
            return ShapeLabel(rot, ref, num, fil, chg, id1, -1, -1)
        elif len(name) == 7:
            id1 = int(name[5])
            id2 = name[6].split('.')[0]
            if 'N' in id2:
                id2 = int(id2.split('N')[0])
            return ShapeLabel(rot, ref, num, fil, chg, id1, id2, -1)
        elif len(name) == 8:
            id1 = int(name[5])
            id2 = int(name[6])
            id3 = name[7].split('.')[0]
            if 'N' in id3:
                id3 = int(id3.split('N')[0])
            return ShapeLabel(rot, ref, num, fil, chg, id1, id2, id3)

    def get_imgs(self, path, imgs):
        idx = 0
        for f in path.iterdir():
            img_path = path / f.name
            imgs.append((self.construct_label(f.name), np.array(Image.open(img_path))))
            idx += 1

    def init_pairs(self):
        self.init_rot_pairs()
        self.init_ref_pairs()
        self.init_num_pairs()
        self.init_chg_pairs()
        self.init_fil_pairs()

    def init_rot_pairs(self):
        test = Path.cwd() / 'test'
        if not test.exists():
            Path.mkdir(test)
        for label, im in self.rot_imgs:
            im_id = str(label.id1) + str(label.id2) + str(label.id3)
            if not im_id in self.rot_dict.keys():
                self.rot_dict[im_id] = [(label, im)]
            else:
                self.rot_dict[im_id].append((label, im))
        t = 0
        for im_id in self.rot_dict.keys():
            for label1, im1 in self.rot_dict[im_id]:
                for label2, im2 in self.rot_dict[im_id]:
                    rotation = (label1.rot - label2.rot) % 180
                    im = np.concatenate((im1, im2), axis=0)
                    label = PairLabel(rotation, 0, label1.num, label1.fil, 0)
                    self.rot_pairs.append((label1, im))
                    im2 = Image.fromarray(np.uint8(im)).convert('L')
                    im2.save("test/test_" + str(t) + '.png')
                    t += 1
            
    def init_ref_pairs(self):
        pass

    def init_num_pairs(self):
        pass

    def init_chg_pairs(self):
        pass

    def init_fil_pairs(self):
        pass

if __name__ == '__main__':
    d = ShapeData()