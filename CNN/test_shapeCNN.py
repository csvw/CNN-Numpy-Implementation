from CNNMIMR import NeuralNetworkMR
from ModelLoaderMRBatch import ModelLoaderMR
import ShapeData
import numpy as np

if __name__ == "__main__":
    nn = NeuralNetworkMR()
    data = ShapeData.ShapeData()
    loss = 0
    cumulative_acc = np.array([0., 0., 0., 0., 0.])
    md_loader = ModelLoaderMR(nn)
    #md_loader.load_model('test_batch_full_running1')
    #md_loader.save_model('sanityCheck')
    batch = True
    if batch:
        for j in range(1000):
            for i, batch in enumerate(data.data_batch):
                _, l, acc = nn.train_batch(batch)
                #nn.grad_check(batch)
                loss += l
                cumulative_acc += acc
                if i % 3 == 0 and i > 0:
                    print(
                        '[Step %d]\tPast 100 Steps: Average Loss %.3f | Rot: %d Ref: %d Num: %d Fil: %d Dif: %d'  %
                        (i+1, loss / 3, cumulative_acc[0], cumulative_acc[1], cumulative_acc[2], cumulative_acc[3], cumulative_acc[4])
                    )
                    loss = 0
                    cumulative_acc = np.array([0., 0., 0., 0., 0.])
                    
                if i > 3:
                    break

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

    
