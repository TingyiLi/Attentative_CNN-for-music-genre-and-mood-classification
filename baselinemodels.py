import keras.layers
from keras.utils import plot_model
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
from keras import regularizers
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import pandas as pd


def buildChoi(width, height, depth,classesGenres,classesValence, param,activate="softmax"):

    regparam = param.regparam
    dropout = param.dropout

    inputShape = (height, width, depth)

    visible = Input(inputShape, name='input1')

    bn0=BatchNormalization(name='BN_input')(visible)
    conv1 = Conv2D(filters=32, kernel_size=3, strides=3, activation='elu',
                   kernel_initializer='glorot_normal',padding='same',name='Conv1')(bn0)
    bn1 = BatchNormalization(name='BN_conv1')(conv1)
    pool1 = MaxPooling2D(pool_size=(1,2), padding='same', name='pool1')(bn1)
    conv2 = Conv2D(filters=128, kernel_size=3, strides=3, activation='elu',
                   kernel_initializer='glorot_normal',padding='same', name='Conv2')(pool1)
    bn2 = BatchNormalization(name='BN_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(1,2), padding='same', name='pool2')(bn2)
    conv3 =  Conv2D(filters=128, kernel_size=3, strides=3, activation='elu',
                   kernel_initializer='glorot_normal',padding='same', name='Conv3')(pool2)
    bn3 = BatchNormalization(name='BN_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(1,2),padding='same', name='pool3')(bn3)

    conv4 =  Conv2D(filters=192, kernel_size=3, strides=3, activation='elu',
                   kernel_initializer='glorot_normal', padding='same', name='Conv4')(pool3)
    bn4 = BatchNormalization(name='BN_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(3,5),padding='same', name='pool4')(bn4)

    conv5 =  Conv2D(filters=256, kernel_size=3, strides=3, activation='elu',
                   kernel_initializer='glorot_normal', padding='same', name='Conv5')(pool4)
    bn5 = BatchNormalization(name='BN_conv5')(conv5)
    pool5 = MaxPooling2D(pool_size=(4,4),padding='same', name='pool5')(bn5)

    flat = Flatten(name='Flatten')(pool5)
    dropd0=keras.layers.Dropout(dropout, name='Drop_Conv4')(flat)


    if param.FCLayers ==1:

        hidden1 = Dense(10, activation='relu',kernel_regularizer=regularizers.l2(regparam), name='FC1')(dropd0)
        bnd1 = BatchNormalization(name='BN_FC1')(hidden1)
        dropd1 = keras.layers.Dropout(dropout, name='DropOut_FC1')(bnd1)
        outputGenre = Dense(classesGenres, activation=activate, name="genreOutput")(dropd1)
        outputValence = Dense(classesValence, activation=activate, name = "valenceOutput")(dropd1)
    else:

        hidden1 = Dense(300, activation='elu',kernel_regularizer=regularizers.l2(regparam))(dropd0)
        bnd1 = BatchNormalization()(hidden1)
        dropd1 = keras.layers.Dropout(dropout)(bnd1)

        hidden2 = Dense(20, activation='elu',kernel_regularizer=regularizers.l2(regparam))(dropd1)
        bnd2 = BatchNormalization()(hidden2)
        dropd2 = keras.layers.Dropout(dropout)(bnd2)
        outputGenre = Dense(classesGenres, activation=activate, name="genreOutput")(dropd2)
        outputValence = Dense(classesValence, activation=activate, name = "valenceOutput")(dropd2)

    modelFunc = Model(inputs=visible, outputs=[outputGenre,outputValence])

    return modelFunc


def buildZhang(width, height, depth,classesGenres,classesValence, param,activate="softmax"):

    regparam = param.regparam
    dropout = param.dropout

    inputShape = (height, width, depth)

    visible = Input(inputShape, name='input1')

    bn0=BatchNormalization(name='BN_input')(visible)
    conv1 = Conv2D(filters=128, kernel_size=1, strides=1, activation='relu',
                   kernel_initializer='glorot_normal',padding='same',name='Conv1')(bn0)
    bn1 = BatchNormalization(name='BN_conv1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,1), padding='same', name='pool1')(bn1)
    conv2 = Conv2D(filters=128, kernel_size=1, strides=1, activation='relu',
                   kernel_initializer='glorot_normal',padding='same', name='Conv2')(pool1)
    bn2 = BatchNormalization(name='BN_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,1), padding='same', name='pool2')(bn2)
    conv3 =  Conv2D(filters=256, kernel_size=1, strides=1, activation='relu',
                   kernel_initializer='glorot_normal',padding='same', name='Conv3')(pool2)
    bn3 = BatchNormalization(name='BN_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(6,1),padding='same', name='pool3Max')(bn3)
    pool4 = AveragePooling2D(pool_size=(6,1),padding='same', name='pool3Ave')(bn3)
    poolave = keras.layers.average([pool4,pool3], name='pool3Combined')

    flat = Flatten(name='Flatten')(poolave)
    dropd0=keras.layers.Dropout(dropout, name='Drop_Conv3')(flat)


    if param.FCLayers ==1:

        hidden1 = Dense(15, activation='relu',kernel_regularizer=regularizers.l2(regparam), name='FC1')(dropd0)
        bnd1 = BatchNormalization(name='BN_FC1')(hidden1)
        dropd1 = keras.layers.Dropout(dropout, name='DropOut_FC1')(bnd1)
        outputGenre = Dense(classesGenres, activation=activate, name="genreOutput")(dropd1)
        outputValence = Dense(classesValence, activation=activate, name = "valenceOutput")(dropd1)
    else:
        hidden1 = Dense(300, activation='relu',kernel_regularizer=regularizers.l2(regparam), name='FC1')(dropd0)
        bnd1 = BatchNormalization(name='BN_FC1')(hidden1)
        dropd1 = keras.layers.Dropout(dropout, name='DropOut_FC1')(bnd1)
        hidden2 = Dense(150, activation='relu',kernel_regularizer=regularizers.l2(regparam), name='FC2')(dropd1)
        bnd2 = BatchNormalization(name='BN_FC2')(hidden2)
        dropd2 = keras.layers.Dropout(dropout, name='DropOut_FC2')(bnd2)

        outputGenre = Dense(classesGenres, activation=activate, name="genreOutput")(dropd2)
        outputValence = Dense(classesValence, activation=activate, name = "valenceOutput")(dropd2)



    modelFunc = Model(inputs=visible, outputs=[outputGenre,outputValence])

    return modelFunc
