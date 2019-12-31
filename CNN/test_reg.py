from NeuralNetworkMultipleRegression import NeuralNetworkMR
import SquareData
from ModelLoaderMR import ModelLoaderMR

if __name__ == "__main__":
    nn = NeuralNetworkMR()
    data = SquareData.SquareData()
    loss = 0
    cumulative_acc = 0
    for i, key in enumerate(data.keys):
        im = data.train_images[key]
        label = data.train_labels[key]
        _, l, acc = nn.train(im, label)
        loss += l
        cumulative_acc += acc
        if i % 25 == 0 and i > 0:
            print(
                '[Step %d]\tPast 100 Steps: Average Loss %.3f | Accuracy: %d%%' %
                (i+1, loss / 100, cumulative_acc)
            )
            loss = 0
            cumulative_acc = 0
        
        if i == 50:
            break
    md_loader = ModelLoaderMR(nn)
    md_loader.save_model('test2')

    
