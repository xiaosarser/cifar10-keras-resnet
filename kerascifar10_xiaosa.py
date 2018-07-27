from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, Input, Flatten
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255
x_train_mean=np.mean(x_train,axis=0)
x_train_var=np.var(x_train,axis=0)
x_test_mean=np.mean(x_test,axis=0)
x_test_var=np.var(x_test,axis=0)

x_train=(x_train-x_train_mean)/(x_train_var)
x_test=(x_test-x_test_mean)/(x_test_var)

input_shape = x_train.shape[1:]
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
#ResNet Block

def normal_CNN(inputshape):
    inputs = Input(shape=inputshape)
    keras.layers.core.Dropout(0.25, noise_shape=None, seed=None)
    x=Conv2D(kernel_size=3,
             strides=1,
             filters=64,
             kernel_initializer=keras.initializers.random_normal(0.0, 0.001),
             bias_initializer=keras.initializers.constant(0))(inputs)
    x=MaxPooling2D(strides=2)(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(kernel_size=3,
               strides=1,
               filters=128,
               kernel_initializer=keras.initializers.random_normal(0.0, 0.001),
               bias_initializer=keras.initializers.constant(0))(x)
    x = MaxPooling2D(strides=2)(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(kernel_size=3,
               strides=1,
               filters=256,
               kernel_initializer=keras.initializers.random_normal(0.0, 0.001),
               bias_initializer=keras.initializers.constant(0))(x)
    x = MaxPooling2D(strides=2)(x)
    x=Activation(activation='relu')(x)
    y = Flatten()(x)
    outputs = Dense(10,
                    activation='softmax',
                    kernel_initializer='he_normal',
                    )(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



def Resnet_block(input,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = input
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnetv1(inputshape):
    inputs=Input(shape=inputshape)
    x = Resnet_block(input=inputs)
    num_filters = 16
    for stack in range(3):
        for res_block in range(3):
            strides = 1
            if stack>0 and res_block==0:
                strides=2
            y=Resnet_block(input=x,kernel_size=3,num_filters=num_filters,strides=strides)
            y=Resnet_block(input=y,kernel_size=3,num_filters=num_filters,activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                x=Resnet_block(x,
                               kernel_size=3,
                               num_filters=num_filters,
                               strides=strides,
                               activation=None,
                               batch_normalization=False
                               )
            x=keras.layers.add([x, y])
            x=Activation('relu')(x)
        num_filters*=2
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(10,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

model=normal_CNN(input_shape)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=200)
score = model.evaluate(x_test, y_test, batch_size=32)

model.save('my_model_new.h5')
t=model.summary()
print(type(t))

