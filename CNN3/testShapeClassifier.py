from CNN import CNNMR
from MLP import MLP
from ShapeData import ShapeData
import numpy as np
from Shapes import Shapes
import random
from PIL import Image
from ModelLoader2 import ModelLoaderMR
from MLPLoader import MLPLoader

if __name__ == "__main__":
    nn             = MLP()
    data           = Shapes()
    cumulative_acc = 0
    loss           = 0
    total_avg      = 0
    prev           = 0
    c              = 0
    m              = 0
    model_loader   = MLPLoader(nn)
    # model_loader.load('shape_classifier_3')
    for j in range(22):
        for i, batch in enumerate(data.data):
            # Image.fromarray(batch[1]).convert('L').show()
            # print(batch[0])
            # print(batch)
            p, l, acc = nn.train(batch)
            m += 1
            c += 1
            loss += l
            cumulative_acc += acc
            total_avg = prev + (l - prev) / m
            prev = total_avg
            # print(p, batch[0], np.argmax(p))
            # print(np.argmax(p), np.argmax(batch[0]), p, p.shape)
            # if i == 5:
            #     input("Press enter to continue")
            if i % (len(data.data) // 25) == 0:
                print(
                    '[Step %d]\tPast %d Steps: Tot_Avg_Loss: %.3f Average Loss %.3f | Acc: %.3f'  %
                    (i+1, len(data.data) // 25, total_avg, loss / c, cumulative_acc / c)
                )
                loss = 0
                cumulative_acc = 0
                c = 0
            # if i == 2:
            #     break
        # random.shuffle(data.data)
    model_loader.save('shape_classifier_1')
