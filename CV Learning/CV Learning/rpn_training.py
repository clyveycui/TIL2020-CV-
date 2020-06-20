import os
import sys
import PIL
import pickle
import json
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
from data_gen import create_ytrue_train, generate_anchor_boxes, preprocess_anchor_boxes

def get_val_data(pickle_file, json_file):
    with open(pickle_file, 'rb') as f:
        imgs_dict = pickle.load(f)
    data_dict = {}
    for imgid in imgs_dict:
        data_dict[imgid] = []
    with open(json_file, 'r') as f:
        annotations_dict = json.load(f)
        annotations_list = annotations_dict['annotations']
    for annotation in annotations_list:
        try:
            img_id = annotation['image_id']
            imwidth = imgs_dict[img_id][1]
            imheight = imgs_dict[img_id][2]
            c = annotation['category_id'] # TODO: make sure that category ids start from 1, not 0. Need to set back_ground as another category
            boxleft,boxtop,boxwidth,boxheight = annotation['bbox']
            box_cenx = boxleft + boxwidth/2.
            box_ceny = boxtop + boxheight/2.
            x,y,w,h = box_cenx/imwidth, box_ceny/imheight, boxwidth/imwidth, boxheight/imheight
            data_dict[img_id].append( [c,x,y,w,h] )
        except KeyError:
            continue
    anchor_boxes = generate_anchor_boxes(480,480,16,scale = [64,128,256], no_exceed_bound = True)
    x, y = [], []
    for img_id, labels in data_dict.items():
        x.append( imgs_dict[img_id][0])
        y.append( create_ytrue_train( imgs_dict[img_id][0], np.array(labels),anchor_boxes, iou_upper=0.7, iou_lower = 0.3 ))
    return np.array(x), np.array(y)

def rpn_regr_loss(ytrue, ypred):
    #y_labels are in the shape [bs, i*j*9, 6]
    mask = tf.where(ytrue != 0, 1., 0.)
    ypred = tf.math.multiply(mask,ypred)
    return keras.losses.Huber()(ytrue,ypred)

def rpn_class_loss(ytrue, ypred):
    cl = tf.reduce_all(ytrue == 0, axis = 2)
    cl2 = tf.stack((cl,cl), axis=-1)
    ypred = tf.where(cl2, 0., ypred)
    return keras.losses.BinaryCrossentropy()(ytrue, ypred)

img_pickle = r'D:\Til data\images.p'
#val_img_pickle = r'/content/images_val.p'
#save_model_folder = r'/content/'
json_annotation = r'D:\Til data\train.json'
#val_annotation = r'/content/val.json'
bs = 16
n_epochs_warmup = 100
n_epochs_after = 100
#train_x, train_y = get_val_data(img_pickle, json_annotation)
#print(train_x.shape)
#print(train_y.shape)
input_shape = (480,480,3)
#save_model_path = os.path.join( save_model_folder, 'rpn_model.h5' )

#Transfers VGG16 layer until last convolutional layer
#model = tf.keras.applications.VGG19(include_top = False, input_shape = (480,480,3))

#model_2 = tf.keras.models.Sequential(name="VGG16")
#for layer in model.layers[:-1]:
#    model_2.add(layer)
#for layer in model_2.layers:
#    layer.trainable = False

#inputs = keras.Input(shape = input_shape)
#x = model_2(inputs)
#x = layers.Conv2D(512, 3, padding="same")(x)
#object_classification = layers.Conv2D(18, 1, padding="same")(x)
#object_classification = layers.Reshape((8100, 2))(object_classification)
#object_classification = layers.Softmax(axis=-1, name="c")(object_classification)
#bbox_regression = layers.Conv2D(36, 1, padding="same")(x)
#bbox_regression = layers.Reshape((8100,4), name="r")(bbox_regression)
#rpn = keras.Model(inputs, outputs=[object_classification, bbox_regression], name="test_model")
#rpn.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#               loss={"c":rpn_class_loss, "r":rpn_regr_loss},
#               loss_weights={"c" : 1./256, "r" : 4*1./900},
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])

#rpn.summary()

#model_plot = tf.keras.utils.plot_model(rpn, to_file ="d:\\programming\\python\\dsta cv\\model3.png",show_shapes=True)
#model_img_f = "d:\\programming\\python\\dsta cv\\model.png"
#keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)

#TRAINING
#file_path = r"D:\Til data\rpn_model.h5"
#rpn = tf.keras.models.load_model(file_path, custom_objects={'custom_loss':custom_loss})
#for layer in rpn.layers:
#    layer.trainable = True
#rpn.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
#            loss=custom_loss,
#            metrics=[tf.keras.metrics.RootMeanSquaredError()])

#rpn.summary()

#val_x, val_y = get_val_data(img_pickle, json_annotation)
#print(val_x.shape)
#print(val_y.shape)
#with open(r"D:\Til data\rpn_train_x.p", "wb") as f:
#    pickle.dump(val_x,f)

#with open(r"D:\Til data\rpn_train_y.p", "wb") as f:
#    pickle.dump(val_y,f)




with open(r"D:\Til data\rpn_train_x.p", "rb") as f:
    val_x = pickle.load(f)

with open(r"D:\Til data\rpn_train_y.p", "rb") as f:
    val_y = pickle.load(f)

class_y = val_y[:,:,:2]
regr_y = val_y[:,:,2:]

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath=r"D:\Til data\rpn\rpn_best.h5",
                                                                save_weights_only=False,
                                                                monitor='val_loss',
                                                                mode='auto',
                                                                save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-8)



rpn.fit(x=val_x,
        y={"c" : class_y, "r" : regr_y},
        batch_size=1,
        epochs=n_epochs_warmup, 
        validation_split = 0.2,
        callbacks=[earlystopping, model_checkpoint_callback, reduce_lr],
        verbose = 1)

del rpn
rpn = tf.keras.models.load_model(r"D:\Til data\rpn\rpn_best.h5", custom_objects = {"rpn_class_loss":rpn_class_loss,"rpn_regr_loss":rpn_regr_loss})
for layer in rpn.get_layer('VGG16').layers:
    layer.trainable = True
rpn.summary()

rpn.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
               loss={"c":rpn_class_loss, "r":rpn_regr_loss},
               loss_weights={"c" : 1./256, "r" : 4*1./900},
               metrics=[tf.keras.metrics.RootMeanSquaredError()])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath=r"D:\Til data\rpn\rpn_finetuned_best.h5",
                                                                save_weights_only=False,
                                                                monitor='val_loss',
                                                                mode='auto',
                                                                save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-8)

rpn.fit(x=val_x,
        y={"c" : class_y, "r" : regr_y}, 
        batch_size=4,
        epochs=50, 
        validation_split = 0.2,
        callbacks=[earlystopping, model_checkpoint_callback, reduce_lr],
        verbose = 1)


rpn.save(r"D:\Til data\rpn\rpn_final.h5")