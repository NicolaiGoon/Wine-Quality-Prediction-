#%%
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

#%%
# read dataset
dataset = pd.read_csv('./data/winequality-combined.csv')

dataset.tail()
dataset.describe().transpose()
dataset.hist()
dataset["quality"].describe()
# data is ok

# %%
train_dataset = dataset.sample(frac=0.8,random_state=1)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_dataset.pop("quality")
test_labels = test_dataset.pop("quality")

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 2])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

# %%
# use gpu
with tf.device('/device:GPU:0'):

    features = np.array(train_features)

    model_normalizer = layers.Normalization(axis=-1)
    model_normalizer.adapt(features)

    model = tf.keras.Sequential(
        [
            model_normalizer,
            layers.Dense(64,activation='relu'),
            layers.Dense(64,activation='relu'),
            layers.Dense(1)
        ]
    )

    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    history = model.fit(
        train_features,
        train_labels,
        epochs=100,
        validation_split = 0.4)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_loss(history)

    model.evaluate(test_features,test_labels)

# %%
