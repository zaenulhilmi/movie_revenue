import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


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


raw_dataset = pd.read_csv('./datasets/movies.csv')
dataset = raw_dataset.copy()


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


train_stats = train_dataset.describe()
train_stats.pop("revenue")
train_stats = train_stats.transpose()


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

train_labels = train_dataset.pop('revenue')
test_labels = test_dataset.pop('revenue')

model = build_model()


EPOCHS = 1000

print(normed_train_data[['popularity', 'budget', 'vote_average', 'vote_count']])
print(train_dataset)


history = model.fit(
  normed_train_data[['popularity', 'budget', 'vote_average', 'vote_count']], train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

loss, mae, mse = model.evaluate(normed_test_data[['popularity', 'budget', 'vote_average', 'vote_count']], test_labels, verbose=2)
print(loss, mae, mse)
test_predictions = model.predict(normed_test_data[['popularity', 'budget', 'vote_average', 'vote_count']]).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [revenue]')
plt.ylabel('Predictions [revenue]')
lims = [0, 1000000000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
