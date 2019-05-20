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
from sklearn.metrics import confusion_matrix
import pandas as pd
from baselinemodels import buildChoi, buildZhang
from CNNmodel import build

def plot_confusion_matrix(cm,
                          target_names,
                          figsizex,
                          figsizey,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(figsizex, figsizey))
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.grid(False)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
    plt.show()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
opt = Adam(lr=param.LR)

loaded_model.compile(loss={'genreOutput': 'categorical_crossentropy', 'valenceOutput': 'categorical_crossentropy'},
              loss_weights={'genreOutput': 1, 'valenceOutput': 1.25}, optimizer=opt,
                     metrics={'genreOutput':'categorical_accuracy', 'valenceOutput':'categorical_accuracy'})

'''Evaluating the model on the test set'''
scores = model.evaluate(testX, [testGenres,testValence], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["genreOutput_loss"], label="genre_train_loss")
plt.plot(np.arange(0, N), H.history["genreOutput_categorical_accuracy"], label="genre_train_acc")
plt.plot(np.arange(0, N), H.history["valenceOutput_loss"], label="valence_train_loss")
plt.plot(np.arange(0, N), H.history["valenceOutput_categorical_accuracy"], label="valence_train_acc")
plt.title("Genre Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["val_genreOutput_categorical_accuracy"], label="val_GenreAcc")
plt.plot(np.arange(0, N), H.history["val_valenceOutput_categorical_accuracy"], label="val_ValenceAcc")
plt.title("Testing Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")

#confusion matriced
pred = model.predict(testX)
cmGenres=confusion_matrix(pred[0].argmax(axis=1), testGenres.argmax(axis=1))
cmValence=confusion_matrix(pred[1].argmax(axis=1), testValence.argmax(axis=1))

plot_confusion_matrix(cmValence,label_encoder_valence.classes_,4,3,'Confusion Matrix Mood', normalize=True)
plot_confusion_matrix(cmGenres,label_encoder.classes_,5,4,'Confusion Matrix Genres', normalize=True)

danceability = featData[:,0]
energy = featData[:,1]
valence = featData[:,2]
accousticness = featData[:,3]

'''Checking out the distribution of valence in the total dataset'''
plt.hist(valence)
