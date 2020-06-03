import os
import PIL
import pickle
import numpy as np
from tqdm import tqdm
from math import log, exp
from random import shuffle
from skimage.transform import resize
from IPython.display import Image, display
from PIL import ImageEnhance, ImageFont, ImageDraw

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.data_utils import Sequence


import img_aug

input_shape = (224,224,3)
#Transfers VGG16 layer until last convolutional layer
model = tf.keras.applications.vgg16.VGG16(input_shape = (224,224,3),weights="imagenet", include_top=False)
model.layers.pop()
model_2 = tf.keras.models.Sequential()
for layer in model.layers[:-1]:
    model_2.add(layer)
for layer in model_2.layers:
    layer.trainable = False

inputs = keras.Input(shape = input_shape)
x = model_2(inputs)
x = layers.Conv2D(512, 3, padding="same")(x)
object_classification = layers.Conv2D(18, 1, padding="same", name="classification")(x)
bbox_regression = layers.Conv2D(36, 1, padding="same", name="regression")(x)
model3 = keras.Model(inputs, outputs=[object_classification, bbox_regression], name="test_model")
model3.compile( optimizer=tf.keras.optimizers.Adam(0.001),
                 loss={'classification' : keras.losses.CategoricalCrossentropy(from_logits=False),
                       'regression' :  keras.losses.Huber()},
                 loss_weights=[1/256, 1/240],
                 metrics=['accuracy'] )
model_plot = tf.keras.utils.plot_model(model3, to_file ="D:\\Programming\\Python\\DSTA CV\\model3.png",show_shapes=True)




#model_img_f = "D:\\Programming\\Python\\DSTA CV\\model.png"
#keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)
