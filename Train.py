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
from baselinemodels import buildChoi, buildZhang
from CNNmodel import build

def ConvertValenceToLabel(data):
    if data<=.3:
        return 'Sad'
    elif (data>.3 and data<=.6):
        return 'moodNeutral'
    else:
        return 'Happy'

class Params():
    """Class that loads hyperparameters from a json file."""
    def __init__(self, json_path):
        self.json_path = json_path

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

xDataRaw = pickle.load(open( 'xDataMelSample.pkl', "rb" ))
genreDataRaw = pickle.load(open( 'genreDataSample.pkl', "rb" ))
featDataRaw = pickle.load(open( 'featDataSample.pkl', "rb" ))

featDataRaw = featDataRaw[:,0:4]

xData = np.asarray(xDataRaw) / 255.0
genreData = np.array(genreDataRaw)
featData= np.array(featDataRaw)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(genreData)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
genres = onehot_encoder.fit_transform(integer_encoded)
print(genres)

danceability = featData[:,0]
energy = featData[:,1]
valence = featData[:,2]
accousticness = featData[:,3]
ValenceData = []
for data in valence:
    ValenceData.extend([ConvertValenceToLabel(data)])

label_encoder_valence = LabelEncoder()
onehot_encoder_valence = OneHotEncoder(sparse=False)
integer_encoded_Valence = label_encoder_valence.fit_transform(ValenceData)
integer_encoded_Valence = integer_encoded_Valence.reshape(len(integer_encoded_Valence), 1)
valenceOneHot = onehot_encoder.fit_transform(integer_encoded_Valence)
print(valenceOneHot)

test_portion = .05
dev_portion = .05
test_size = round(len(xData)*test_portion)
dev_size = round(len(xData)*dev_portion)
randTotal= np.random.randint(1,len(xData),test_size+dev_size)
randTest = randTotal[0:test_size]
randDev = randTotal[test_size:]
origIndices= list(range(0, len(xData)))
trainIndices = [x for x in origIndices if x not in randTotal]

devX = xData[randDev]
devGenres =genres[randDev]
devValence =valenceOneHot[randDev]

trainX = xData[trainIndices]
trainGenres =genres[trainIndices]
trainValence =valenceOneHot[trainIndices]

testX = xData[randTest]
testGenres = genres[randTest]
testValence = valenceOneHot[randTest]


testX=testX.reshape(testX.shape[0],testX.shape[1],testX.shape[2],1)
trainX=trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2],1)
devX = devX.reshape(devX.shape[0],devX.shape[1],devX.shape[2],1)


param = Params('params.json')
param.dict['LR']=.001
param.dict['BatchSize']=150
param.dict['regparam']=.001
param.dict['dropout'] = .5
param.dict['FCLayers']=1
EPOCHS = 75
print(label_encoder_valence.classes_)
print(label_encoder.classes_)

IMAGE_DIMS = trainX[0].shape

INIT_LR = param.LR
BS = param.BatchSize

print("[INFO] compiling model...")
model = build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
              depth=IMAGE_DIMS[2], classesGenres = len(label_encoder.classes_), classesValence = len(label_encoder_valence.classes_)
              ,param=param, activate="softmax")

opt = Adam(lr=param.LR)

model.compile(loss={'genreOutput': 'categorical_crossentropy', 'valenceOutput': 'categorical_crossentropy'},
              loss_weights={'genreOutput': 1, 'valenceOutput': 1.25}, optimizer=opt,
                     metrics={'genreOutput':'categorical_accuracy', 'valenceOutput':'categorical_accuracy'})

# checkpoint
#filepath="weights.best.hdf5"
filepath="weightsCheckpoint.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_valenceOutput_categorical_accuracy',
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

H = model.fit(trainX,
	{"genreOutput": trainGenres, "valenceOutput": trainValence},
	validation_data=(devX,
		{"genreOutput": devGenres, "valenceOutput": devValence}),
	epochs=EPOCHS,callbacks=callbacks_list, verbose=1)
#callbacks=callbacks_list,
# save the model to disk
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#model.summary()
