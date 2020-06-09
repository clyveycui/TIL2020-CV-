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
from tensorflow.keras.layers import Layer


import img_aug
import data_gen


def custom_loss(ytrue, ypred):
    #y_labels are in the shape [bs, i*j*9, 6]
    class_w = 1.0/256
    balance_factor = 10
    regr_w = 1.0/2400 * balance_factor


    ypred_class = ypred[:,:,:2]
    ypred_regr = ypred[:,:,2:]
    ytrue_class = ytrue[:,:,:2]
    ytrue_regr = ytrue[:,:,2:]
    ypred_class = tf.where(tf.stack((tf.reduce_all(ytrue_class == 0, axis = 2), tf.reduce_all(ytrue_class == 0, axis = 2)), axis = -1), 0., ypred_class)
    if (ytrue_regr.shape[0]!= None):
        for b in range(ytrue_regr.shape[0]):
            for i in range(ytrue_regr.shape[1]):
                if ytrue_class[b,i,0] != 1:
                    ypred_regr[b,i,0] = 0
                    ypred_regr[b,i,1] = 0
                    ypred_regr[b,i,2] = 0
                    ypred_regr[b,i,3] = 0

    class_loss = keras.losses.BinaryCrossentropy()(ytrue_class, ypred_class)
    regr_loss = keras.losses.Huber()(ytrue_regr, ypred_regr)
    return class_w * class_loss + regr_w * regr_loss

img_folder = r'D:\til2020\train\train'
val_img_folder = r'D:\til2020\val\val'
save_model_folder = r'D:\til2020'
json_annotation = r'D:\til2020\train.json'
val_annotation = r'D:\til2020\val.json'
bs = 16
n_epochs_warmup = 2
n_epochs_after = 2
train_sequence = data_gen.TILSequence(img_folder,json_annotation,bs,img_aug.aug_default,testmode = False)
val_sequence = data_gen.TILSequence(val_img_folder, val_annotation, bs, img_aug.aug_identity,testmode = False)
input_shape = (640,960,3)
save_model_path = os.path.join( save_model_folder, 'rpn_model.h5' )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath=save_model_path,
                                                                save_weights_only=False,
                                                                monitor='val_loss',
                                                                mode='auto',
                                                                save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)

#Transfers VGG16 layer until last convolutional layer
model = tf.keras.applications.vgg16.VGG16(input_shape = (640,960,3),weights="imagenet", include_top=False)
model.layers.pop()
model_2 = tf.keras.models.Sequential(name="VGG16")
for layer in model.layers[:-1]:
    model_2.add(layer)
for layer in model_2.layers:
    layer.trainable = False

inputs = keras.Input(shape = input_shape)
x = model_2(inputs)
x = layers.Conv2D(512, 3, padding="same")(x)
object_classification = layers.Conv2D(18, 1, padding="same", name="classification")(x)
object_classification = layers.Reshape((21600, 2))(object_classification)
object_classification = layers.Softmax(axis=-1)(object_classification)
bbox_regression = layers.Conv2D(36, 1, padding="same", name="regression")(x)
bbox_regression = layers.Reshape((21600,4))(bbox_regression)
pred_concat = layers.Concatenate()([object_classification,bbox_regression])
rpn = keras.Model(inputs, outputs=pred_concat, name="test_model")
rpn.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss=custom_loss,
               metrics=['accuracy'])
model_plot = tf.keras.utils.plot_model(rpn, to_file ="D:\\Programming\\Python\\DSTA CV\\model3.png",show_shapes=True)

#model_img_f = "D:\\Programming\\Python\\DSTA CV\\model.png"
#keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)

#TRAINING

rpn.fit(x=train_sequence, 
        epochs=n_epochs_warmup, 
        validation_data=val_sequence, 
        callbacks=[model_checkpoint_callback, earlystopping, reduce_lr])



load_model_path = os.path.join( save_model_folder, 'rpn_model.h5' )
del rpn
rpn = tf.keras.models.load_model(load_model_path, custom_objects={'custom_loss':custom_loss})
#for layer in rpn.get_layer('VGG16').layers:
#    layer.trainable = True
#model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#                                                                filepath=save_model_path,
#                                                                save_weights_only=False,
#                                                                monitor='val_loss',
#                                                                mode='auto',
#                                                                save_best_only=True)
#earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)
#rpn.fit(x=train_sequence, 
#          epochs=n_epochs_after, 
#          validation_data=val_sequence, 
#          callbacks=[model_checkpoint_callback, earlystopping, reduce_lr])

rpn.save(os.path.join( save_model_folder, 'rpn_model_final.h5' ))



#APPLY get_rp then nms on the output of rpn
#Feed that into a roi pooling layer
