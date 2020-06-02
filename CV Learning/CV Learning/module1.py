import tensorflow as tf
from tensorflow import keras
import numpy as np
  

ytrue = np.array([1, 1, 0, 0.5, 0.5, 1, 1])
ypred = np.array([0.5, 0.9, 0, 1, 1, 1, 1])
ypred = tf.where(ypred[0] > 0.5)
category_loss = tf.keras.losses.CategoricalCrossentropy() ( ytrue[1:3],)
print(category_loss)