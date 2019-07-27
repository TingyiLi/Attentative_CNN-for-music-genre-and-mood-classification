# -*- coding: utf-8 -*-
"""
Created on Wed May 1st 5 10:02:35 2019

@author: Tingyi Li
"""
import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import json_lines
import urllib.request
#from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import json
import spotipy
import time
import sys
from pathlib import Path
from slugify import slugify

client_id=''#private
client_secret=''
credentials = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret)

token = credentials.get_access_token()
spotify = spotipy.Spotify(auth=token)

tracks = pd.read_csv('tracks.csv')
genres = pd.read_csv('genres.csv')


lsgenres =genres.loc[genres['top_level'].unique()].sort_values('#tracks', ascending=False).dropna().title.tolist()
numGenres = len(lsgenres)
mygenres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

def GetFilePathName(trackid, smallormedium):
    audio_dir = r'C:\Users\Tingyi_Li\projects'
    folder = 'fma_'+smallormedium
    filename = str('{:06}'.format(int(gid))+'.mp3')
    audio_dir = os.path.join(audio_dir,  folder, filename[:3])
    fullpath = os.path.join(audio_dir, filename)
    return fullpath

numPerGenre = 75
datainfo = []

for g in mygenres:
    count = 0
    print(g)
    small = (tracks['subset'] == 'small') | (tracks['subset'] == 'medium')
    genre = tracks['genre_top']==g
    filterred = tracks.loc[small & genre, ('track_id','title.1','name','genre_top', 'subset')]
    gids = tracks.loc[small & genre, ('track_id')].tolist()
    for gid in gids:
        row = filterred.loc[filterred['track_id']==gid]
        item ={}
        item['genre']= row.genre_top.values[0]
        item['artist'] = row.name.values[0]
        item['title'] = row["title.1"].values[0]
        search_str = item['artist']+' '+item['title']
        results = spotify.search(search_str)
        if len(results['tracks']['items'])==0:
            print ('no spotify result for '+search_str)
            continue
        print(results['tracks']['items'][0]['name'])
        uri = results['tracks']['items'][0]['uri']
        popularity = results['tracks']['items'][0]['popularity']
        features = spotify.audio_features(uri)
        if len(features)==0:
            print('no featuers found for '+search_str)
            continue
        feature=features[0]
        item['valence']=feature['valence']
        item['danceability']=feature['danceability']
        item['energy']=feature['energy']
        item['instrumentalness']=feature['instrumentalness']
        item['liveness']=feature['liveness']
        item['loudness']=feature['loudness']
        item['speechiness']=feature['speechiness']
        item['acousticness']=feature['acousticness']
        itempath = GetFilePathName(gid, row['subset'].values[0])
        item['path']=itempath
        testfile = Path(itempath)
        if(not(testfile.is_file())):
            print('file not found for '+str(itempath))
            continue
        datainfo.extend([item])
        count=count+1
        print(search_str+' '+g+' '+str(count))
        if count>=numPerGenre:
            break
r

with open('totaldata.json', 'w') as fp:
    json.dump(datainfo, fp)
