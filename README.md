# CNN-Numpy-Implementation
Manually implemented a convolutional neural network without using modern libraries such as pytorch and tensorflow. In other words, I built a neural network from scratch which involved implementing both forward and backpropagation. I coded backpropagation by hand, manually implementing the partial derivatives of every layer using numpy. An example of the equations I utilized can be found here.

Bendersky, E. (October 28, 2016). The Softmax function and its derivative. Retrieved from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

My most sophisticated and successful network was a multiple regression CNN that tracked several features. It could correctly determine the difference in the number of shapes between two images and the presence or absence of reflection, but it had trouble measuring the degree of rotation and the difference in fill color between shapes. In order to train this network, I had to generate my own dataset.

## Project Description

The Raven's Progressive Matrices (RPM) visual IQ test has been used by AI researchers as a testbed for developing new strategies, algorithms, and cognitive agents. By building agents that can solve progressively more difficult problems, researchers hope to push the state of the art in artificial intelligence. An example RPM problem is displayed below.

![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/Basic%20Problem%20B-10.PNG)

In CS 7637, Knowledge-Based Artificial Intelligence, students are tasked with developing solutions to RPM problems. Since this is an open research problem, students are encouraged to explore novel strategies and dive into the literature on the subject. The cognitive agent is developed in three stages over the course of the semester, with students solving more difficult problems in each successive stage.

I decided that I would like to utilize deep learning to incorporate visual comprehension in my agent. Students were not allowed to use external libraries such as tensorflow or pytorch, and so I would have to implement the neural network from scratch. Furthermore, I only had my own computer on which to perform training, and this can be a prohibitive constraint.

I used this as an opportunity to gain more exposure to deep learning. A complete list of the materials I referenced in this project can be found in the appendix at the bottom of this readme.

Some example images from my shape generator are shown below.

![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc2.png)
![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc3.png)
![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc4.png)

The image below depicts my network training on the mnist dataset. The classification accuracy score on the right increases and levels off at around 90%.

![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc5.png)

For those who are interested, I provide an overview of my development process below.

## Development

My first challenge was to get a basic network up and running. It took me several days of chasing tutorials and thumbing through derivations of different components of the backpropagation algorithm before I was able to get my first network running on the mnist dataset. Once I had a basic network, I started to look for ways to adapt it into a multiple regression network. I decided that I would try to use a multiple regression output with mean squared loss to predict five features when given an input image consisting of two figures concatenated in numpy (for example, A to B or A to C in the Raven’s matrix). I would train my CNN to predict a vector of features (rotation, reflection, number, fill, shape difference) and then obtain predictions for A to B and A to C. Then I would generate predictions between B and a possible answer and C and a possible answer for every answer among the options. Finally, I would subtract the corresponding vector pairs, and then sum their absolute value. The figure that gave me the lowest difference would be selected as the answer. (To be clear, this is the difference between two differences, AtoB - CtoAns, where AtoB and CtoAns are both differences between the network's output for the figures).

However, in order to train my CNN, I would need data. I spent a good deal of time writing my own shape generator in python. It took me several more days to both write the data generator and finally to get this version of the CNN working. Implementing backpropagation was very difficult. And my implementation only partially worked: I couldn’t get its prediction accuracy for rotation below an average miss of twenty degrees. It could have been a flaw either with my CNN implementation or with my data generator (perhaps I generated bad image-label pairs). Other features were more accurate: for the shape count, it would miss by one twenty percent of the time.

After searching around for a bit, I thought that perhaps I could get better accuracy if I implemented batch normalization. I spent several days trying to get it to work, and was left with a broken codebase (I neglected to create a git repository for this project).

After this, I thought that a simpler CNN solution might work. If I could train my neural network to determine whether two transformations were identical, then I could use that to find the answer figure with the most similar transformations to the transformations in the matrix. To make the solution more robust, instead of taking the raw categorical output, I could take the probability outputs from softmax for the row prediction and the column prediction and sum them. This would give me the transformation pair with the highest total probability.

I spent much of the rest of my time before the deadline rewriting my network--twice. I finally got it working on mnist again, but ultimately had to abandon it in order to make a simple solution that would meet the basic performance requirements for part one of the RPM project sequence.

## Appendix

I've included below a list of resources I consulted while developing my neural network.

Resources

Ivanov, S. (2019, April 16). 37 Reasons why your Neural Network is not working. Retrieved from https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607.

Stack Exchange user sjishan (2017, Febuary 10). Neural Network for Multiple Output Regression Retrieved from https://datascience.stackexchange.com/questions/16890/neural-network-for-multiple-output-regression?rq=1.

Nielsen, & A., M. (1970, January 1). Neural Networks and Deep Learning. Retrieved from http://neuralnetworksanddeeplearning.com/chap2.html.

ML Cheat Sheet. Backpropagation. (n.d.). Retrieved from https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html.

Ng, A. (n. d.). Improving Neural Networks: Hyperparameter tuning, Regularization and Optimization. Fitting batch norm into a neural network. Retrieved from https://www.coursera.org/learn/deep-neural-network/lecture/RN8bN/fitting-batch-norm-into-a-neural-network

Karpathy, A. (2016, December 19). Yes you should understand backprop. Retrieved from https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b.

Zakka, K. (n. d.). Deriving the Gradient for the Backward Pass of Batch Normalization. Retrieved from https://kevinzakka.github.io/2016/09/14/batch_normalization/.

Dahal, P. (2017, May 17). BatchNorm Layer - Understanding and eliminating Internal Covariance Shift. Retrieved from https://deepnotes.io/batchnorm.

Bendersky, E. (October 28, 2016). The Softmax function and its derivative. Retrieved from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

Dahal, P. (2017, May 28). Classification and Loss Evaluation - Softmax and Cross Entropy Loss. Retrieved from https://deepnotes.io/softmax-crossentropy.

Victor Zhou. (2019, August 8). CNNs, Part 1: An Introduction to Convolutional Neural Networks. Retrieved from https://victorzhou.com/blog/intro-to-cnns-part-1/

Victor Zhou. (2019, August 8). CNNs, Part 2: Training a Convolutional Neural Network. Retrieved from https://victorzhou.com/blog/intro-to-cnns-part-2/.

Dahal, P. (2017, May 7). Convolution Layer - The core idea behind CNNs. Retrieved from https://deepnotes.io/convlayer.

Dahal, P. (2017, May 5). Introduction to Convolutional Neural Networks. Retrieved from https://deepnotes.io/intro.

StackOverflow user neel (2017, May 15). ValueError: could not broadcast input array from shape (224,224,3) into shape (224,224). Retrieved from https://stackoverflow.com/questions/43977463/valueerror-could-not-broadcast-input-array-from-shape-224-224-3-into-shape-2.

Brownlee, J. (2019, October 3). How to Fix the Vanishing Gradients Problem Using the ReLU. Retrieved from https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/.

StackExchange user Danny. (2015, September 12). Neural network backpropagation with RELU. Retrieved from https://stackoverflow.com/questions/32546020/neural-network-backpropagation-with-relu.

In Convolutional Nets (CNN) isn't the maxpool layer and ReLU layer redundant? (n.d.). Retrieved from https://www.quora.com/In-Convolutional-Nets-CNN-isnt-the-maxpool-layer-and-ReLU-layer-redundant.

Dahal, P. (2017, May 28). Classification and Loss Evaluation - Softmax and Cross Entropy Loss. Retrieved from https://deepnotes.io/softmax-crossentropy.

StackOverflow user DilithiumMatrix. (2015, November 6). numpy max vs amax vs maximum. Retrieved from https://stackoverflow.com/questions/33569668/numpy-max-vs-amax-vs-maximum.

StackOverflow user blaz. (2015, December 7). Difference between numpy dot() and Python 3.5+ matrix multiplication @. Retrieved from https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication.

StackExchange user Jonathan DEKHTIAR. (2017, September 20). Why convolutions always use odd-numbers as filter_size. Retrieved from https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size.

StackExchange user Tendero. (2017, July 15). How to update filter weights in CNN? Retrieved from https://stats.stackexchange.com/questions/291708/how-to-update-filter-weights-in-cnn

Rathi, M. (n.d.). Backpropagation in a Convolutional Neural Network. Retrieved from https://mukulrathi.com/demystifying-deep-learning/conv-net-backpropagation-maths-intuition-derivation/.

Agarwal, M. (2017, December 18). Back Propagation in Convolutional Neural Networks - Intuition and Code. Retrieved from https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199.

StackExchange user koryakinp. (2018, Febuary 6). back propagation in CNN. Retrieved from https://datascience.stackexchange.com/questions/27506/back-propagation-in-cnn.

StackOverflow user JohnAllen. (2016, October 5). Possible explanations for loss increasing? Retrieved from https://stackoverflow.com/questions/39868939/possible-explanations-for-loss-increasing.

StackOverflow user Devin Haslam. (2018, Febuary 3). Cross entropy loss suddenly increases to infinity. Retrieved from https://stackoverflow.com/questions/48600374/cross-entropy-loss-suddenly-increases-to-infinity.

Escontrela, A. (2018, June 17). Convolutional Neural Networks from the ground up. Retrieved from https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1.

THE MNIST DATABASE. (n.d.). Retrieved from http://yann.lecun.com/exdb/mnist/.

Bendersky, E. (2018, May 22) Backpropagation through a fully-connected layer. Retrieved from https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/.

Ng, A. (n. d.). Neural Networks and Deep Learning. Gradient descent for Neural Networks. Retrieved from https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks.

Ng, A. (n. d.). Neural Networks and Deep Learning. Vectorizing Logistic Regression's Gradient Output. Retrieved from https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks.

Ng, A. (n. d.). Improving Deep Neural Networks: Hyperparameter tuning, Regularization, and Optimization. Normalizing activations in a network. Retrieved from https://www.coursera.org/learn/deep-neural-network/lecture/4ptp2/normalizing-activations-in-a-network.

Ng, A. (n. d.). Convolutional Neural Networks. One Layer of a Convolutional Network. Retrieved from https://www.coursera.org/learn/deep-neural-network/lecture/4ptp2/normalizing-activations-in-a-network.

Rai, S. (2017, December 24). Forward And Backpropagation in Convolutional Neural Network. Retrieved from https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e.

Hallström, E. (2016, December 5). Backpropagation from the beginning. Retrieved from https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d.
