import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# print(tf.__version__)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

# column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
#                 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv('./datasets/tmdb_5000_movies.csv')

dataset = raw_dataset.copy()
dataset.tail()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["budget", "revenue", "vote_count"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("revenue")
train_stats = train_stats.transpose()

print(train_stats['mean'])

# normed_train_data = norm(train_dataset)
# normed_test_data = norm(test_dataset)
# print(normed_train_data)

train_labels = train_dataset.pop('revenue')
test_labels = test_dataset.pop('revenue')

model = build_model()

# model.summary()


example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
print(example_result)








