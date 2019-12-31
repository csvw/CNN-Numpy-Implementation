from DataGenerator import DataGenerator

class DataLabel:
    def __init__(self):
        self.gen = DataGenerator()
        self.shape_labels = {}
        self.init_shape_labels()

    def init_shape_labels(self):
        for i, shape in enumerate(self.gen.all_shapes):
            self.shape_labels[shape] = i
    
    def label_shape(self, shapes):
        label = 0
        exp = 0
        for shape in shapes:
            label += (10 ** exp) * self.shape_labels[shape]
        return label

    def label(self, rot, ref, num, fill, changed):
        return [rot, ref, num, fill, changed]
        

    