import keras
import tensorflow as tf
moxel = keras.models.Sequential()
layer = keras.layers.MultiHeadAttention(1,4)

moxel.add(tf.keras.Input(shape=(1,1,1)),tf.keras.Input(shape=(1,1,1)))
moxel.add(layer)