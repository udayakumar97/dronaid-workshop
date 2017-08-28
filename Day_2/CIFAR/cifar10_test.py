# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
K.set_image_dim_ordering('th')

seed=7
numpy.random.seed(seed)

#load data
(image_train,label_train),(image_test,label_test)=cifar10.load_data()

#normalize inputs from 0-255 to 0.0 -1.0 ?
#image_train=image_train.astype('float32')
image_test=image_test.astype('float32')
#image_train=image_train/255.0
image_test=image_test/255.0

#one hot encode outputs ?
#label_train=np_utils.to_categorical(label_train)
label_test=np_utils.to_categorical(label_test)
num_classes=label_test.shape[1]

#Load the saved model files
loaded_model=load_model('keras_cifar10_trained_model.h5')
print(loaded_model.summary())
score=loaded_model.evaluate(image_test, label_test, verbose=1) # Evaluate the trained model on the test set!
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
