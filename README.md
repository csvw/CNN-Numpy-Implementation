# CNN-Numpy-Implementation
Manually implemented a convolutional neural network without using modern libraries such as pytorch and tensorflow. I built this CNN with the aim of learning more about the low level details of deep neural networks.

## Project Description

The Raven's Progressive Matrices (RPM) visual IQ test has been used by AI researchers as a testbed for developing new strategies, algorithms, and cognitive agents. By building agents that can solve progressively more difficult problems, researchers hope to push the state of the art in artificial intelligence. An example RPM problem is displayed below.

![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/Basic%20Problem%20B-10.PNG)

In CS 7637, Knowledge-Based Artificial Intelligence, students are tasked with developing solutions to RPM problems. Since this is an open research problem, students are encouraged to explore novel strategies and dive into the literature on the subject. The cognitive agent is developed in three stages over the course of the semester, with students solving more difficult problems in each successive stage.

When I was first assigned this project, I decided that I would like to utilize deep learning to incorporate a degree of visual comprehension in my agent. This was a risky decision for several reasons. First, students were not allowed to use external libraries such as tensorflow or pytorch. If I wanted to use deep learning, I would have to implement the neural network completely from scratch. Second, I did not possess an large amount of previous exposure to deep learning. There was no guaruntee that I would be able to successfully implement a neural network. Third, even if I did manage to implement a CNN, it was possible I wouldn't be able to adapt it to the task at hand--that it would provide no benefit to my agent. Finally, the time constaints on deep learning can be problematic. I only had my own computer on which to perform training, and this can be a prohibitive constraint.

Despite these risks, I decided to go through with the approach. I had been looking for a way to gain more exposure to deep learning, and decided that the requirement for a manual implementation would force me to learn more about the mechanics of deep learning than I otherwise would have. Looking back, this assessment was correct--if you would like to get a sense of what it took for me to get my network up and running with my limited background, I would encourage you to scroll to the bottom of this readme and take a look at my appendix, which includes a list of the references I consulted while building my network.

I ultiately did get a basic CNN working. It trains and correctly classifies simple datasets, such as the mnist dataset and a shapes dataset I generated algorithmically. However, when I attempted to use it to classify more sophisticated relationships between pairs of images, it floundered.

My most sophisticated and successful network was not sufficiently accurate for reliable use in my project. It was a multiple regression CNN that tracked several features. It could correctly determine the difference in the number of shapes between two images and the presence or absence of reflection, but it had trouble measuring the degree of rotation and the difference in fill color between shapes. In order to train this network, I had to generate my own dataset. Some example images from my shape generator are shown below.


![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc2.png)
![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc3.png)
![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc4.png)

The image below depicts my network training on the mnist dataset. The classification accuracy score on the right increases and levels off at around 90%.

![alt text](https://github.com/csvw/CNN-Numpy-Implementation/blob/master/sc5.png)

For those who are interested, I provide a brief overview of my development process below.

## Development

My first challenge was to get a basic network up and running. It took me several days of chasing tutorials and thumbing through derivations of different components of the backpropagation algorithm before I was able to get my first network running on the mnist dataset. Once I had a basic network, I started to look for ways to adapt it into a multiple regression network. I decided that I would try to use a multiple regression output with mean squared loss to predict five features, given an input image consisting of two figures concatenated in numpy (for example, A to B or A to C in the Raven’s matrix). My idea was that I would train my CNN to predict a vector of features (rotation, reflection, number, fill, shape difference) and then get predictions for A to B and A to C. Then I would get predictions between B and a possible answer and C and a possible answer for every answer among the options. Finally, I would subtract the corresponding vector pairs, and then sum their absolute value. The figure that gave me the lowest difference would be selected as the answer. (To be clear, this is the difference between two differences, AtoB - CtoAns, where AtoB and CtoAns are both differences between the networks output for the figures).

However, in order to train my CNN, I would need data. I spent a good deal of time writing my own shape generator in python. It took me several more days to both write the data generator and finally to get this version of the CNN working. Implementing backpropagation was very difficult. And my implementation only partially worked: I couldn’t get its prediction accuracy for rotation below an average miss of twenty degrees. It could have been a flaw either with my CNN implementation or with my data generator (perhaps I generated bad image-label pairs). Other features were more accurate: for the shape count, it would miss by one twenty percent of the time.

After searching around for a bit, I thought that perhaps I could get better accuracy if I implemented batch normalization. I spent several days trying to get it to work, and was left with a broken codebase (I neglected to create a git repository for this project).

I thought that a simpler CNN solution might work. If I could train my neural network to determine whether two transformations were identical, then I could use that to find the answer figure with the most similar transformations to the transformations in the matrix. To make the solution more robust, instead of taking the raw categorical output, I could take the probability outputs from softmax and sum them. This would give me the transformation pair with the highest total probability.

I spent most of the rest of my time before the deadline rewriting my network--twice. I finally got it working on mnist again, but ultimately had to abandon it in order to make a simple solution that would meet the basic performance requirements for part one of the RPM project sequence.

## Appendix

I've included below a list of resources I consulted while developing my neural network.

Resources

https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

https://datascience.stackexchange.com/questions/16890/neural-network-for-multiple-output-regression?rq=1

https://stats.stackexchange.com/questions/261227/neural-network-for-multiple-output-regression

http://neuralnetworksanddeeplearning.com/chap2.html

https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html

https://www.coursera.org/learn/deep-neural-network/lecture/RN8bN/fitting-batch-norm-into-a-neural-network

https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

https://kevinzakka.github.io/2016/09/14/batch_normalization/

https://deepnotes.io/batchnorm

https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/

https://deepnotes.io/softmax-crossentropy

https://victorzhou.com/blog/intro-to-cnns-part-1/

https://victorzhou.com/blog/intro-to-cnns-part-2/

https://deepnotes.io/convlayer

https://deepnotes.io/intro

https://stackoverflow.com/questions/43977463/valueerror-could-not-broadcast-input-array-from-shape-224-224-3-into-shape-2

https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html

https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/

https://stackoverflow.com/questions/32546020/neural-network-backpropagation-with-relu

https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html

https://www.quora.com/In-Convolutional-Nets-CNN-isnt-the-maxpool-layer-and-ReLU-layer-redundant

https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html

https://deepnotes.io/softmax-crossentropy

https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.rot90.html

https://stackoverflow.com/questions/33569668/numpy-max-vs-amax-vs-maximum

https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication

https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html

https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size

https://stats.stackexchange.com/questions/291708/how-to-update-filter-weights-in-cnn

https://mukulrathi.com/demystifying-deep-learning/conv-net-backpropagation-maths-intuition-derivation/

https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199

https://datascience.stackexchange.com/questions/27506/back-propagation-in-cnn

https://stackoverflow.com/questions/39868939/possible-explanations-for-loss-increasing

https://stackoverflow.com/questions/48600374/cross-entropy-loss-suddenly-increases-to-infinity

https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html

https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

http://dilab.gatech.edu/publications/Kunda%20McGreggor%20Goel%202011%20AAAI.pdf

https://pdfs.semanticscholar.org/11b5/490215bc9442467f7b1a5fc11db8f87b47de.pdf

https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1

http://yann.lecun.com/exdb/mnist/

https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/

https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks

https://www.coursera.org/learn/neural-networks-deep-learning/lecture/IgFnJ/vectorizing-logistic-regressions-gradient-output

https://www.coursera.org/learn/deep-neural-network/lecture/4ptp2/normalizing-activations-in-a-network

https://www.coursera.org/learn/convolutional-neural-networks/lecture/nsiuW/one-layer-of-a-convolutional-network

https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

https://stackoverflow.com/questions/11937985/how-to-use-pil-python-image-library-rotate-image-and-let-black-background-to

https://codegolf.stackexchange.com/questions/109917/draw-plot-a-heart-graph

https://www.cs.princeton.edu/courses/archive/fall08/cos429/lecture_linear_filters_edge_detection.pdf

https://www.microimages.com/documentation/TechGuides/81FiltEdge.pdf

https://web.cs.wpi.edu/~emmanuel/courses/cs545/S14/slides/lecture04.pdf

https://en.wikipedia.org/wiki/Kernel_(image_processing)

https://datascience.stackexchange.com/questions/16890/neural-network-for-multiple-output-regression

https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d

http://neuralnetworksanddeeplearning.com/chap2.html

https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110

https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/

https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

https://datascience.stackexchange.com/questions/19272/deep-neural-network-backpropogation-with-relu

https://www.quora.com/How-do-I-fix-exploding-and-vanishing-gradients-How-do-ReLUs-LSTMs-and-new-techniques-like-batchnorm-help-with-these-problems

https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
