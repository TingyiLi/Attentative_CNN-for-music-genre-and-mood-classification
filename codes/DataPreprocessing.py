# -*- coding: utf-8 -*-
"""
Created on Wed May 1st 5 10:02:35 2019

@author: Tingyi Li
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
#import cv2
import os
import json_lines
import json
from PIL import Image
import cv2
import librosa
import librosa.display
import CreateSpectrogramLibrosa as specrosa
def ConvertDataToLabels(data):
    return data['genre']

def ConvertDataToFeats(data):
    labels =[]
    labels.extend([data['danceability']])
    labels.extend([data['energy']])
    labels.extend([data['valence']])
    labels.extend([data['acousticness']])
    labels.extend([data['loudness']])
    return labels


def ClassifyHighLow(strFeature, doubleRating):
    if(doubleRating<.5):
        return 'low_'+strFeature
    else:
        return 'high_'+strFeature

def ClassifyHighLow(strFeature, doubleRating):
    if(doubleRating<.5):
        return 'low_'+strFeature
    else:
        return 'high_'+strFeature


xDataList = []
genreData = []
featData=[]
drWavs= r'..\Tingyi_Li\projects\wavs'
drImgs= r'..\Tingyi_Li\projects\imgs'
dataJl = r'..\Tingyi_Li\projects\totaldata.json'

#loop over files in wav folder
directory = os.fsencode(drWavs)
with open(dataJl) as json_file:
    jData = json.load(json_file)

''' Make the imgpath valid,about 100 samples missing cuz of this'''
count =0
nslices=10
for j in jData:
    print(j['genre'])
    filepath = j['path']
    count=count+1
    labels = ConvertDataToLabels(j)
    feats = ConvertDataToFeats(j)

    y, sr = librosa.load(filepath, mono=True, duration=30)
    songslices = []
    for i in range(nslices):
        songslices.append(y[i*65024:(i+1)*65024])
    for Slice in songslices:
        # Make Mel spectrogram
        S = librosa.feature.melspectrogram(Slice, sr=sr, n_mels=128)
        # Convert to log scale (dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        xDataList.extend([log_S])
        genreData.extend([labels])
        featData.extend([feats])



xDataSmaller=[]
genreDataSmaller=[]
featDataSmaller=[]
pergenre = 75
BluesCount =0
ClassicalCount=0
CountryCount=0
DiscoCount=0
HiphopCount=0
JazzCount=0
MetalCount=0
PopCount=0
ReggaeCount=0
RockCount=0
for n in range(0, len(xDataList)):
    if(genreData[n]=='Blues' and BluesCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        RockCount = RockCount+1
    elif (genreData[n]=='Classical' and ClassicalCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        PopCount = PopCount+1
    elif (genreData[n]=='Country' and CountryCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        HipHopCount = HipHopCount+1
    elif (genreData[n]=='Disco' and DiscoCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        FolkCount = FolkCount+1
    elif (genreData[n]=='Hip-hop' and HiphopCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        InstrumentalCount = InstrumentalCount+1
    elif (genreData[n]=='Jazz' and JazzCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        ElectronicCount = ElectronicCount+1
    elif (genreData[n]=='Metal' and MetalCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        ElectronicCount = ElectronicCount+1
    elif (genreData[n]=='Pop' and PopCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        ElectronicCount = ElectronicCount+1
    elif (genreData[n]=='Reggae' and ReggaeCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        ElectronicCount = ElectronicCount+1
    elif (genreData[n]=='Rock' and RockCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        ElectronicCount = ElectronicCount+1



for xdata in xDataList:
    if (xdata.shape != xDataList[0].shape):
        print(xdata.shape)

with open('xDataMel.pkl', 'wb') as f:
    pickle.dump(np.asarray(xDataList), f)

with open('genreData.pkl', 'wb') as f:
    pickle.dump(np.asarray(genreData), f)

with open('featData.pkl', 'wb') as f:
    pickle.dump(np.asarray(featData), f)

with open('xDataMelSmaller2.pkl', 'wb') as f:
    pickle.dump(np.asarray(xDataSmaller), f)

with open('genreDataSmaller2.pkl', 'wb') as f:
    pickle.dump(np.asarray(genreDataSmaller), f)

with open('featDataSmaller2.pkl', 'wb') as f:
    pickle.dump(np.asarray(featDataSmaller), f)
