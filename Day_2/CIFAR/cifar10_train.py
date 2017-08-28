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
image_train=image_train.astype('float32')
image_test=image_test.astype('float32')
image_train=image_train/255.0
image_test=image_test/255.0

#one hot encode outputs ?
label_train=np_utils.to_categorical(label_train)
label_test=np_utils.to_categorical(label_test)
num_classes=label_test.shape[1]

#Model for CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(image_train, label_train, validation_data=(image_test, label_test), epochs=epochs, batch_size=16)
# Final evaluation of the model
scores = model.evaluate(image_test, label_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model_name = 'keras_cifar10_trained_model.h5'
model.save(model_name)
print('Saved trained model as %s ' % model_name)
