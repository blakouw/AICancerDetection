import keras
import matplotlib.pyplot as plt
from keras import regularizers
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from prepare_images import read_and_prepare
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from prepare_images import read_and_prepare

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Input, regularizers
import keras_tuner

X, Y = read_and_prepare()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

def tune(hp):
    model = Sequential()
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=24, max_value=48, step=8),
        kernel_size=(3, 3),
        activation='relu',
        strides=(1, 1),
        input_shape=(50, 50, 3)
        ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(
        filters=hp.Int('conv_2_filter', min_value=48, max_value=72, step=8),
        kernel_size=(3, 3),
        activation='relu',
        strides=(1, 1)
        ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(
        filters=hp.Int('conv_3_filter', min_value=96, max_value=136, step=8),
        kernel_size=(3, 3),
        activation='relu',
        strides=(1, 1)
        ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_1_units', min_value=48, max_value=128, step=16),
        activation='relu'
    ))
    model.add(Dense(
        units=hp.Int('dense_2_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0, max_value=0.4, step=0.1)))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

tuner = keras_tuner.RandomSearch(
        tune,
        objective='val_loss',
        max_trials=10
    )
tuner.search(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
best_model = tuner.get_best_models()[0]
best_model.summary()
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)