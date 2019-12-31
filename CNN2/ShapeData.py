import numpy as np
from PIL import Image
from pathlib import Path
import random

class ShapeLabel:
    def __init__(self, rot, ref, num, fil, changed, id1, id2, id3, id4):
        self.rot = rot
        self.ref = ref
        self.num = num
        self.fil = fil
        self.chg = changed
        self.id1 = id1
        self.id2 = id2
        self.id3 = id3
        self.id4 = id4

    def to_array(self):
        return np.array([self.rot, self.ref, self.num, self.fil, self.chg])


class PairLabel:
    def __init__(self, rot, ref, num, fil, changed):
        self.rot = rot
        self.ref = ref
        self.num = num
        self.fil = fil
        self.chg = changed


class ShapeData:
    def __init__(self):
        self.cwd      = Path.cwd() / 'DLData'
        self.rot_path = self.cwd / 'rotations'
        self.ref_path = self.cwd / 'reflections'
        self.num_path = self.cwd / 'numbers'
        self.shp_path = self.cwd / 'shapes'
        self.rot_imgs = list()
        self.ref_imgs = list()
        self.num_imgs = list()
        self.shp_imgs = list()
        self.get_imgs(self.rot_path, self.rot_imgs)
        self.get_imgs(self.ref_path, self.ref_imgs)
        self.get_imgs(self.num_path, self.num_imgs)
        self.get_shp_imgs(self.shp_path, self.shp_imgs)
        self.rot_dict  = {}
        self.ref_dict  = {}
        self.num_dict  = {}
        self.rot_pairs = []
        self.ref_pairs = []
        self.num_pairs = []
        self.fil_pairs = []
        self.chg_pairs = []
        self.data_batch = []
        self.data = []
        self.shape_data = []
        self.shape_data_batch = []
        self.init_pairs()
        self.shuffle_indices = [i for i in range(10000)]
        self.shuffled = []
        self.shuffle(67)
        self.init_data_batch(self.shuffled)

    
    def shuffle(self, seed):
        self.shuffled = []
        random.seed(seed)
        while len(self.shuffle_indices) > 0:
            indices_idx = random.randint(0, len(self.shuffle_indices)-1)
            idx = self.shuffle_indices[indices_idx]
            self.shuffle_indices.remove(idx)
            self.shuffled.append(idx)
        
    
    def construct_label(self, name):
        name = name.split('_')
        rot = int(name[0])
        ref = int(name[1])
        num = int(name[2])
        fil = int(name[3])
        chg = int(name[4])
        if len(name) == 6:
            id1 = int(name[5].split('.')[0])
            return ShapeLabel(rot, ref, num, fil, chg, id1, -1, -1, -1)
        elif len(name) == 7:
            id1 = int(name[5])
            id2 = name[6].split('.')[0]
            if 'N' in id2:
                id2 = int(id2.split('N')[0])
            return ShapeLabel(rot, ref, num, fil, chg, id1, id2, -1, -1)
        elif len(name) == 8:
            id1 = int(name[5])
            id2 = int(name[6])
            id3 = name[7].split('.')[0]
            if 'N' in id3:
                id3 = int(id3.split('N')[0])
            return ShapeLabel(rot, ref, num, fil, chg, id1, id2, id3, -1)
        elif len(name) == 9:
            id1 = int(name[5])
            id2 = int(name[6])
            id3 = int(name[7])
            id4 = name[8].split('.')[0]
            if 'N' in id4 and not (id4 == 'N' or id4 == 'N2'):
                id4 = int(id4.split('N')[0])
            elif 'N' == id4 or id4 == 'N2':
                return ShapeLabel(rot, ref, num, fil, chg, id1, id2, id3, -1)
            return ShapeLabel(rot, ref, num, fil, chg, id1, id2, id3, id4)

    def construct_shp_label(self, name):
        n = name.split('_')
        label = [int(n[0]), int(n[1])]
        return label

    def get_shp_imgs(self, path, imgs):
        idx = 0
        for f in path.iterdir():
            img_path = path / f.name
            imgs.append((self.construct_shp_label(f.name), np.array(Image.open(img_path))))
            idx += 1

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
            im_id = str(label.id1) + str(label.id2) + str(label.id3) + str(label.fil)
            if not im_id in self.rot_dict.keys():
                self.rot_dict[im_id] = [(label, im)]
            else:
                self.rot_dict[im_id].append((label, im))
        t = 0
        for im_id in self.rot_dict.keys():
            for label1, im1 in self.rot_dict[im_id]:
                for label2, im2 in self.rot_dict[im_id]:
                    rotation = -(label1.rot - label2.rot)
                    im = np.concatenate((im1, im2), axis=0)
                    label = PairLabel(rotation, 0, label1.num, label1.fil, 0)
                    self.rot_pairs.append((label1, im))
                    if t < 200:
                        im2 = Image.fromarray(np.uint8(im)).convert('L')
                        im2.save("test/test_" + str(t) + '_' + str(rotation) + '.png')
                        t += 1
        print("Rot pairs: " + str(len(self.rot_pairs)))
            
    def init_ref_pairs(self):
        test = Path.cwd() / 'testref'
        if not test.exists():
            Path.mkdir(test)
        for label, im in self.ref_imgs:
            im_id = str(label.id1) + str(label.id2) + str(label.id3) + str(label.fil) + str(label.rot)
            if not im_id in self.ref_dict.keys():
                self.ref_dict[im_id] = [(label, im)]
            else:
                self.ref_dict[im_id].append((label, im))
        t = 0
        for im_id in self.ref_dict.keys():
            cur = 0
            for label1, im1 in self.ref_dict[im_id]:
                if label1.ref == 0:
                    cur = (label1, im1)
            for label2, im2 in self.ref_dict[im_id]:
                if cur == (label2, im2):
                    continue
                reflection = label2.ref
                im = np.concatenate((cur[1], im2), axis=0)
                label = PairLabel(0, reflection, label2.num, label2.fil, 0)
                self.ref_pairs.append((label2, im))
                if t < 200:
                    im2 = Image.fromarray(np.uint8(im)).convert('L')
                    im2.save("testref/test_" + str(t) + '_' + str(reflection) + '.png')
                    t += 1
        print(len(self.ref_pairs))

    def init_num_pairs(self):
        #print(self.num_imgs[0][0].id1)
        test = Path.cwd() / 'testnum'
        if not test.exists():
            Path.mkdir(test)
        t = 0
        count = 0
        for label1, im1 in self.num_imgs:
            if label1 is None:
                count += 1
                #print(count)
        #print(len(self.num_imgs))
        for label1, im1 in self.num_imgs:
            for label2, im2 in self.num_imgs:
                num = label2.num - label1.num
                dif = 0
                same_primary_shape = label1.id1 == label2.id1
                same_secondary_shape = label1.id2 == label2.id2
                if same_primary_shape:
                    if same_secondary_shape:
                        dif = 0
                    else:
                        self.compute_c(True, label2.num, label1.num)
                else:
                    if same_secondary_shape:
                        self.compute_c(False, label2.num, label1.num)
                    else:
                        dif = min(label1.num, label2.num)
                
                #print(label1.id1, label2.id1, label1.id2, label2.id2, label1.num, label2.num, num, dif)    
                fill_dif = label2.fil - label1.fil
                im = np.concatenate((im1, im2), axis=0)
                label = PairLabel(0, 0, num, fill_dif, dif)
                self.num_pairs.append((label2, im))
                if t < 200:
                    im2 = Image.fromarray(np.uint8(im)).convert('L')
                    im2.save("testnum/test_" + str(t) + '_' + str(num) + '_' + str(fill_dif) + '_' + str(dif) + '.png')
                    t += 1
        print(len(self.num_pairs))

    def compute_c(self, e, n, t):
        if e:
            result = 0
            for i in range(1, n+1):
                if i <= t:
                    if i % 2 == 0:
                        result += 1
                else:
                    if i % 2 == 1:
                        result -= 1
            return result
        else:
            result = 0
            for i in range(1, n+1):
                if i <= t:
                    if i % 2 == 1:
                        result += 1
                else:
                    if i % 2 == 0:
                        result -= 1
            return result

    def init_chg_pairs(self):
        pass

    def init_fil_pairs(self):
        pass

    def init_shapes(self):
        self.shuffle_indices = [i for i in range(len(self.shp_imgs))]
        self.shuffled = []
        self.shuffle(67)
        for i in range(len(self.shp_imgs)):
            self.shape_data += [self.shp_imgs[self.shuffled[i]]]
        
        for i in range(0, len(self.shp_imgs), 1):
            imgs   = []
            labels = []
            for img in self.shape_data[i:i+1]:
                assert(img[0] is not None)
                imgs.append(img[1])
                labels.append(img[0])
            tensor = np.stack(imgs, axis=0)
            labels = np.stack(labels, axis=0)
            batch = [labels, tensor]
            self.shape_data_batch.append(batch)

    def init_data_batch(self, indices):
        for i in indices:
            self.data += [self.rot_pairs[i]] + [self.ref_pairs[i % 1500]] + [self.num_pairs[i]]
        print("Img: " + str(self.rot_pairs[0][1].shape))

        for i in range(0, 16 * 1200, 16):
            imgs   = []
            labels = []
            for img in self.data[i:i+16]:
                imgs.append(img[1])
                labels.append(img[0].to_array())
            tensor = np.stack(imgs, axis=2)
            labels = np.stack(labels, axis=1)
            batch = [labels, tensor]
            self.data_batch.append(batch)
        # print("Batch:")
        # print(self.data_batch[0][0].shape)
        # print(self.data_batch[0][1].shape)
        self.init_shapes()



if __name__ == '__main__':
    d = ShapeData()
