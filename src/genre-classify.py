import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Activation
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import utils as np_utils
import pickle

meta = pd.read_csv("movie.metadata.tsv", sep='\t', header=None)

meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

plots = []

with open("plot_summaries.txt", 'r', encoding="utf8") as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)

movie_id = []
plot = []
maxLen = 500
for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

meta['movie_id'] = meta['movie_id'].astype(str)

movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on='movie_id')

genres = []

for i in movies['genre']:
    val = list(json.loads(i).values())
    if len(val) > 0:
         genres.append(val[0])
    else:
        genres.append(val)

movies['genre_new'] = genres
print(movies.shape)
movies = movies[~(movies['genre_new'].str.len() == 0)]
print(movies.shape)


# Getting the fields needed for analysis

text = movies['plot']
labels = movies["genre_new"]

x = text.values[:5000]
y = labels.values[:5000]

# Tokenizing the x axis data
tk = Tokenizer(num_words= 200, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(x)
x = tk.texts_to_sequences(x)
x = pad_sequences(x, maxlen=maxLen)

unique_labels = np.unique(y)

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

y = np_utils.to_categorical(y, num_classes=len(unique_labels))

np.random.seed(200)
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]



index_from=3
start_char = 1
if start_char is not None:
        x = [[start_char] + [w + index_from for w in x1] for x1 in x]
elif index_from:
        x = [[w + index_from for w in x1] for x1 in x]

num_words = None
if not num_words:
        num_words = max([max(x1) for x1 in x])

oov_char = 2
skip_top = 0

if oov_char is not None:
        x = [[w if (skip_top <= w < num_words) else oov_char for w in x1] for x1 in x]
else:
        x = [[w for w in x1 if (skip_top <= w < num_words)] for x1 in x]


x_train, y_train = np.array(x), np.array(y)

x_train = pad_sequences(x_train, maxlen=maxLen)

print(x_train.shape)

max_features = 1000
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxLen))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(len(unique_labels)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=150)

model.save('model')

with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(labelencoder_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)