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

def get_anchor_points(w,h,stride):
    #Takes w, h of image as int or floats 
    #takes stride as int/float

    #returns p*q*2 np array containing (x,y) float coordinates of all anchor points on the image
    #p, q are the number of anchor points along the x axis and y axis respectively

    padx = int(((w*1./stride)%1)*stride)
    pady = int(((h*1./stride)%1)*stride)
    no_x = int(w/stride)
    no_y = int(h/stride)
    aps = np.zeros((no_x, no_y, 2))
    pad_l = padx//2 + (padx%2)
    pad_t = pady//2 + (pady%2)
    cur_x = pad_l+stride//2
    cur_y = pad_t+stride//2
    for i in range(no_x):
        for j in range(no_y):
            aps[i,j] = np.array([cur_x, cur_y])
            cur_y += stride
        cur_x += stride
        cur_y = pad_l+stride//2
    return aps

def generate_anchor_boxes(w, h, stride, scale=[128,256,512], ratio=[0.5, 1, 2], no_exceed_bound = False):
    #w, h, stride for get_anchor_points()
    #scale is the pixel length of the side of a square anchor box, must be a list
    #ratio is the width/height ratio of the anchor boxes, cannot be negative
    #if no_exceed_bound is true, all boxes that exceeds the image are not added in, leaving a 0 value at their supposed locations

    #returns a p*q*(s*r)*4 array, where the anchor box is stored in the form of [x_centre, y_centre, width, height].
    # s, r are the length of the scale and ratio lists respectively

    anchor_points = get_anchor_points(w,h,stride)
    a_shape = anchor_points.shape
    anchor_boxes = np.zeros((a_shape[0], a_shape[1], len(ratio)*len(scale), 4))
    
    side_lengths = []
    for s in scale:
        for r in ratio:
            side_lengths.append([s*np.sqrt(r),s/np.sqrt(r)])
    for i in range(a_shape[0]):
        for j in range(a_shape[1]):
            for k, sl in enumerate(side_lengths) :
                a_box = np.array([anchor_points[i,j][0], anchor_points[i,j][1], sl[0], sl[1]])
                if(no_exceed_bound):
                    if(a_box[0] - a_box[2]/2<0 or a_box[0] + a_box[2]/2>w):
                        continue
                    elif(a_box[1] - a_box[3]/2<0 or a_box[1] + a_box[3]/2>h):
                        continue
                    else:
                        anchor_boxes[i,j,k] = a_box
                else:
                    anchor_boxes[i,j,k] = a_box
    return anchor_boxes



#model_img_f = "D:\\Programming\\Python\\DSTA CV\\model.png"
#keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)
