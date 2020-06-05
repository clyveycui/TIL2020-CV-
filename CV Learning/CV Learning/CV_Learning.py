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

def custom_loss(ytrue, ypred):
    #y_labels are in the shape [bs, i, j, 18+36]
    class_w = 1.0/256
    balance_factor = 10
    regr_w = 1.0/2400 * balance_factor


    ypred_class = ypred[:,:,:,:18]
    ypred_regr = ypred[:,:,:,18:]
    ytrue_class = ytrue[:,:,:,:18]
    ytrue_regr = ytrue[:,:,:,18:]
    ypred_class = tf.where(ytrue_class !=-1, ypred_class, -1)
    if (ytrue_regr.shape[0]!= None):
        for b in range(ytrue_regr.shape[0]):
            for i in range(ytrue_regr.shape[1]):
                for j in range(ytrue_regr.shape[2]):
                    for n in range(ytrue_regr.shape[3]//4):
                        if ytrue_class[b,i,j,2*n] != 1:
                            ypred_regr[b,i,j,4*n] = 0
                            ypred_regr[b,i,j,4*n+1] = 0
                            ypred_regr[b,i,j,4*n+2] = 0
                            ypred_regr[b,i,j,4*n+3] = 0

    class_loss = keras.losses.BinaryCrossentropy()(ytrue_class, ypred_class)
    regr_loss = keras.losses.Huber()(ytrue_regr, ypred_regr)
    return class_w * class_loss + regr_w * regr_loss


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

pred_concat = layers.Concatenate()([object_classification,bbox_regression])

model3 = keras.Model(inputs, outputs=pred_concat, name="test_model")

model3.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss=custom_loss,
               metrics=['accuracy'])
model_plot = tf.keras.utils.plot_model(model3, to_file ="D:\\Programming\\Python\\DSTA CV\\model3.png",show_shapes=True)




#model_img_f = "D:\\Programming\\Python\\DSTA CV\\model.png"
#keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)
