# http://euler.stat.yale.edu/~tba3/stat665/lectures/lec16/notebook16.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28**2).astype('float32') / 255
X_test = X_test.reshape(10000, 28**2).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, input_shape=(28 * 28,)))
model.add(Activation("sigmoid"))
model.add(Dense(10))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd, metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=2,
          verbose=1, validation_split=0.1)

y_hat = model.predict_classes(X_test)

print("\nTeste classification rate %0.05f" % model.evaluate(X_test, Y_test)[0])

pd.crosstab(y_hat, y_test)


