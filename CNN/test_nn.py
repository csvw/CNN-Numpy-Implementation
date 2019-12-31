import NeuralNetwork
import mnist

if __name__ == "__main__":
    nn = NeuralNetwork.NeuralNetwork()
    nn.load_model("test_NN_model1s.csv")
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(mnist.train_images(), mnist.train_labels())):
        l, acc = nn.train(im, label)
        loss += l
        num_correct += acc
        if i % 99 == 0 and i > 0:
            print(
                '[Step %d]\tPast 100 Steps: Average Loss %.3f | Accuracy: %d%%' %
                (i+1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0
        
        if i == 500:
            break
    nn.save_model("test_NN_model1s.csv")