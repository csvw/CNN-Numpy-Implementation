from CNN import CNNMR
from MLP import MLP
from ModelLoader2 import ModelLoaderMR
import ShapeData
import Shapes
import numpy as np
import mnist

if __name__ == "__main__":
    # nn = CNNMR()
    nn = CNNMR()
    data = ShapeData.ShapeData()
    shapes = Shapes.Shapes()
    loss = 0
    md_loader = ModelLoaderMR(nn)
    #md_loader.load('cnn_shape_classifier_f')
    #md_loader.save_model('sanityCheck')
    train = mnist.train_images()
    label = mnist.train_labels()
    shape = True
    if not shape:
        cumulative_acc = np.array([0., 0., 0., 0., 0.])
        for j in range(1000):
            for i, label, train in enumerate(zip(label, train)):
                #nn.test(batch)
                batch = (label, train)
                p, l, acc = nn.train(batch)
                #nn.grad_check(batch)
                loss += l
                cumulative_acc += acc
                
                if i % 99 == 0 and i > 0:
                    print(
                        '[Step %d]\tPast 100 Steps: Average Loss %.3f | Rot: %d Ref: %d Num: %d Fil: %d Dif: %d'  %
                        (i+1, loss / 3, cumulative_acc[0], cumulative_acc[1], cumulative_acc[2], cumulative_acc[3], cumulative_acc[4])
                    )
                    loss = 0
                    cumulative_acc = np.array([0., 0., 0., 0., 0.])
                    
                if i > 300:
                    break
    else:
        cumulative_acc = 0
        total_avg = 0
        prev = 0
        m = 0
        for i in range(1):
            print(shapes.data[i][0])
        for j in range(1):
            loss = 0
            cumulative_acc = 0
            c = 0
            for i, (lab, tra) in enumerate(shapes.data):
                # lab = [1 if i == lab else 0 for i in range(10)]
                batch = [lab, tra]
                #nn.test(batch)
                # if j == 0:
                #     batch = nn.normalize_batch(batch)
                p, l, acc = nn.train(batch)
                #nn.grad_check(batch)
                m += 1
                c += 1
                loss += l
                cumulative_acc += acc
                total_avg = prev + (l - prev) / m
                prev = total_avg
                # print(p)
                # print(lab)
                # print(p)
                # print(l)
                if i % 99 == 0:
                    print(
                        '[Step %d]\tPast 100 Steps: Tot_Avg_Loss: %.3f Average Loss %.3f | Acc: %.3f'  %
                        (i+1, total_avg, loss / c, cumulative_acc / c)
                    )
                    loss = 0
                    cumulative_acc = 0
                    c = 0
                
                # if i > 15:
                #     break
                    
                
    md_loader.save('cnn_shape_classifier_f2')
    # md_loader.save('test_nan')

    #else:
        # for j in range(1):
        #     for i, (label, im) in enumerate(data.data):
        #         label = label.to_array()
        #         _, l, acc = nn.train(label, im)
        #         loss += l
        #         cumulative_acc += acc
        #         if i % 100 == 0 and i > 0:
        #             print(
        #                 '[Step %d]\tPast 100 Steps: Average Loss %.3f | Rot: %d Ref: %d Num: %d Fil: %d Dif: %d'  %
        #                 (i+1, loss / 100, cumulative_acc[0], cumulative_acc[1], cumulative_acc[2], cumulative_acc[3], cumulative_acc[4])
        #             )
        #             loss = 0
        #             cumulative_acc = [0, 0, 0, 0, 0]
            
            
    #md_loader.save_model('test_batchnorm')

    
