#Load Dataset
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import  toimage

#load data
(image_train,label_train),(image_test,label_test)=cifar10.load_data()
#create a grid of 3x3 images
for i in range(0,9):
    pyplot.subplot(330 + 1 +i)
    pyplot.imshow(toimage(image_train[i]))

pyplot.show()