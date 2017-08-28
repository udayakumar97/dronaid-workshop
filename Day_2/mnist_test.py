import os
from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense # the two types of neural network layer we will be using
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import load_model

#num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)
(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data

#X_train = X_train.reshape(num_train, height * width) # Flatten data to 1D
X_test = X_test.reshape(num_test, height * width) # Flatten data to 1D
#X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

#Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

#Load the model and weights
loaded_model=load_model('keras_mnist_trained_model.h5')
print(loaded_model.summary())
score=loaded_model.evaluate(X_test, Y_test, verbose=1) # Evaluate the trained model on the test set!
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))