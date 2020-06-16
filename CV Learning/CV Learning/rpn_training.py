import os
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

    x, y = [], []
    for img_id, labels in data_dict.items():
        x.append( imgs_dict[img_id][0])
        y.append( create_ytrue_train( imgs_dict[img_id][0], np.array(labels), iou_upper=0.7, iou_lower = 0.3 ))
    return np.array(x), np.array(y)

def custom_loss(ytrue, ypred):
    #y_labels are in the shape [bs, i*j*9, 6]
    class_w = 1.0/256
    balance_factor = 10
    regr_w = 1.0/2400 * balance_factor
    ypred_class = ypred[:,:,:2]
    ypred_regr = ypred[:,:,2:]
    ytrue_class = ytrue[:,:,:2]
    ytrue_regr = ytrue[:,:,2:]
    cl = tf.reduce_all(ytrue_class == 0, axis = 2)
    cl2 = tf.stack((cl,cl), axis=-1)
    ypred_class = tf.where(cl2, 0., ypred_class)
    r = tf.where(ytrue_class[:,:,0] == 1, 1., 0.)
    r2 = tf.stack((r,r), axis = -1)
    ypred_regr_mask = tf.concat((r2,r2), axis = -1)
    ypred_regr = tf.math.multiply(ypred_regr_mask, ypred_regr)

    class_loss = keras.losses.BinaryCrossentropy()(ytrue_class, ypred_class)
    regr_loss = keras.losses.Huber()(ytrue_regr, ypred_regr)
    return class_w * class_loss + regr_w * regr_loss

img_pickle = r'D:\Til data\images_train_1000.p'
#val_img_pickle = r'/content/images_val.p'
#save_model_folder = r'/content/'
json_annotation = r'D:\Til data\train.json'
#val_annotation = r'/content/val.json'
bs = 16
n_epochs_warmup = 10
n_epochs_after = 100
#train_x, train_y = get_val_data(img_pickle, json_annotation)
#print(train_x.shape)
#print(train_y.shape)
input_shape = (480,480,3)
#save_model_path = os.path.join( save_model_folder, 'rpn_model.h5' )

#Transfers VGG16 layer until last convolutional layer
#model = tf.keras.applications.vgg16.VGG16(input_shape = (480,480,3),weights="imagenet", include_top=False)
#model.layers.pop()
#model_2 = tf.keras.models.Sequential(name="VGG16")
#for layer in model.layers[:-1]:
#    model_2.add(layer)
#for layer in model_2.layers:
#    layer.trainable = False

#inputs = keras.Input(shape = input_shape)
#x = model_2(inputs)
#x = layers.Conv2D(512, 3, padding="same")(x)
#object_classification = layers.Conv2D(18, 1, padding="same", name="classification")(x)
#object_classification = layers.Reshape((8100, 2))(object_classification)
#object_classification = layers.Softmax(axis=-1)(object_classification)
#bbox_regression = layers.Conv2D(36, 1, padding="same", name="regression")(x)
#bbox_regression = layers.Reshape((8100,4))(bbox_regression)
#pred_concat = layers.Concatenate()([object_classification,bbox_regression])
#rpn = keras.Model(inputs, outputs=pred_concat, name="test_model")
#rpn.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#               loss=custom_loss,
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])



#model_plot = tf.keras.utils.plot_model(rpn, to_file ="d:\\programming\\python\\dsta cv\\model3.png",show_shapes=True)
#model_img_f = "d:\\programming\\python\\dsta cv\\model.png"
#keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)

#TRAINING
file_path = r"D:\Til data\rpn_model.h5"
rpn = tf.keras.models.load_model(file_path, custom_objects={'custom_loss':custom_loss})
for layer in rpn.layers:
    layer.trainable = True
rpn.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=custom_loss,
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

rpn.summary()

#val_x, val_y = get_val_data(img_pickle, json_annotation)
#with open(r"D:\Til data\train_1000_x.p", "wb") as f:
#    pickle.dump(val_x,f)

#with open(r"D:\Til data\train_1000_y.p", "wb") as f:
#    pickle.dump(val_y,f)




with open(r"D:\Til data\train_x.p", "rb") as f:
    val_x = pickle.load(f)

with open(r"D:\Til data\train_y.p", "rb") as f:
    val_y = pickle.load(f)

rpn.fit(x=val_x,
        y=val_y, 
        batch_size=1,
        epochs=n_epochs_warmup, 
        validation_split = 0.2,
        verbose = 2)


#for layer in rpn.get_layer('VGG16').layers:
#    layer.trainable = True

#rpn.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
#               loss=custom_loss,
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])

#rpn.summary()
#rpn.fit(x=train_x,
#        y=train_y, 
#        batch_size=16,
#        epochs=n_epochs_warmup, 
#        validation_split = 0.2,
#        verbose = 2)

rpn.save(r"D:\Til data\rpn_model_finetuned_2.h5")