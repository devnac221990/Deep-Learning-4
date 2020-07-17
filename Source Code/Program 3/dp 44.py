# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

tf.keras.backend.image_data_format()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



# Create the model
model = Sequential()

model.add(Conv2D(32, padding= 'same', kernel_size=3,activation='relu',input_shape=(32,32,3),kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Conv2D(32,padding= 'same', kernel_size=3,activation='relu',input_shape=(32,32,3), kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Conv2D(64,kernel_size=3,padding= 'same',activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128,kernel_size=3,padding= 'same' , activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Conv2D(128,kernel_size=5,padding='same',activation='relu', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save('./model' + '.h5')
history_dict = history.history
history_dict.keys()

# VALIDATION LOSS curves

plt.clf()
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# VALIDATION ACCURACY curves

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, (len(history_dict['accuracy']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print("Test Accuracy: %.2f%%" % (scores[1]*100))

img_idx = 177
plt.imshow(X_test[img_idx],aspect='auto')
plt.show()
print('Actual label:',[np.argmax(y_test[img_idx])])
# Prepare the image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))
img_idx = 187
plt.imshow(X_test[img_idx],aspect='auto')
plt.show()
print('Actual label:',[np.argmax(y_test[img_idx])])
# Prepare the image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))
img_idx = 197
plt.imshow(X_test[img_idx],aspect='auto')
plt.show()
print('Actual label:',[np.argmax(y_test[img_idx])])
# Prepare the image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))
img_idx = 179
plt.imshow(X_test[img_idx],aspect='auto')
plt.show()
print('Actual label:',[np.argmax(y_test[img_idx])])
# Prepare the image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))