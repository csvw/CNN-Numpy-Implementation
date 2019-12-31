from CNN2 import CNNMR
from ModelLoader2 import ModelLoaderMR
import ShapeData
import numpy as np

if __name__ == "__main__":
    nn = CNNMR()
    data = ShapeData.ShapeData()
    loss = 0
    md_loader = ModelLoaderMR(nn)
    #md_loader.load_model('test_batch_full_running1')
    #md_loader.save_model('sanityCheck')
    shape = True
    if not shape:
        cumulative_acc = np.array([0., 0., 0., 0., 0.])
        for j in range(1000):
            for i, batch in enumerate(data.data_batch):
                #nn.test(batch)
                _, l, acc = nn.train(batch)
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
                    
                if i > 3:
                    break
    else:
        cumulative_acc = 0
        total_avg = 0
        prev = 0
        m = 0
        for j in range(500):
            loss = 0
            cumulative_acc = 0
            c = 0
            for i, batch in enumerate(data.shape_data_batch):
                #nn.test(batch)
                _, l, acc = nn.train(batch)
                #nn.grad_check(batch)
                m += 1
                c += 1
                loss += l
                cumulative_acc += acc
                total_avg = prev + (l - prev) / m
                prev = total_avg
                if i % 99 == 0:
                    print(
                        '[Step %d]\tPast 100 Steps: Tot_Avg_Loss: %.3f Average Loss %.3f | Acc: %.3f'  %
                        (i+1, total_avg, loss / c, cumulative_acc / c)
                    )
                    loss = 0
                    cumulative_acc = 0
                    c = 0
                
                    
                
    md_loader.save('testShape1')

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

    
