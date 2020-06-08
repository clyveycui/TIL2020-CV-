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
import data_gen_no_sequence as data_gen


##Source : https://gist.github.com/Jsevillamol/0daac5a6001843942f91f2a3daea27a7
#class ROIPoolingLayer(Layer):

#    """ Implements Region Of Interest Max Pooling 

#        for channel-first images and relative bounding box coordinates

        

#        # Constructor parameters

#            pooled_height, pooled_width (int) -- 

#              specify height and width of layer outputs

        

#        Shape of inputs

#            [(batch_size, pooled_height, pooled_width, n_channels),

#             (batch_size, num_rois, 4)]

           

#        Shape of output

#            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    

#    """

#    def __init__(self, pooled_height, pooled_width, **kwargs):

#        self.pooled_height = pooled_height

#        self.pooled_width = pooled_width

        

#        super(ROIPoolingLayer, self).__init__(**kwargs)

        

#    def compute_output_shape(self, input_shape):

#        """ Returns the shape of the ROI Layer output

#        """

#        feature_map_shape, rois_shape = input_shape

#        assert feature_map_shape[0] == rois_shape[0]

#        batch_size = feature_map_shape[0]

#        n_rois = rois_shape[1]

#        n_channels = feature_map_shape[3]

#        return (batch_size, n_rois, self.pooled_height, 

#                self.pooled_width, n_channels)



#    def call(self, x):

#        """ Maps the input tensor of the ROI layer to its output

        

#            # Parameters

#                x[0] -- Convolutional feature map tensor,

#                        shape (batch_size, pooled_height, pooled_width, n_channels)

#                x[1] -- Tensor of region of interests from candidate bounding boxes,

#                        shape (batch_size, num_rois, 4)

#                        Each region of interest is defined by four relative 

#                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1



#            # Output

#                pooled_areas -- Tensor with the pooled region of interest, shape

#                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)



#        """

#        def curried_pool_rois(x): 

#          return ROIPoolingLayer._pool_rois(x[0], x[1], 

#                                            self.pooled_height, 

#                                            self.pooled_width)

        

#        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)



#        return pooled_areas

    

#    @staticmethod

#    def _pool_rois(feature_map, rois, pooled_height, pooled_width):

#        """ Applies ROI pooling for a single image and varios ROIs

#        """

#        def curried_pool_roi(roi): 

#          return ROIPoolingLayer._pool_roi(feature_map, roi, 

#                                           pooled_height, pooled_width)

        

#        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)

#        return pooled_areas

    

#    @staticmethod

#    def _pool_roi(feature_map, roi, pooled_height, pooled_width):

#        """ Applies ROI pooling to a single image and a single region of interest

#        """



#        # Compute the region of interest        

#        feature_map_height = int(feature_map.shape[0])

#        feature_map_width  = int(feature_map.shape[1])

        

#        h_start = tf.cast(feature_map_height * roi[0], 'int32')

#        w_start = tf.cast(feature_map_width  * roi[1], 'int32')

#        h_end   = tf.cast(feature_map_height * roi[2], 'int32')

#        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')

        

#        region = feature_map[h_start:h_end, w_start:w_end, :]

        

#        # Divide the region into non overlapping areas

#        region_height = h_end - h_start

#        region_width  = w_end - w_start

#        h_step = tf.cast( region_height / pooled_height, 'int32')

#        w_step = tf.cast( region_width  / pooled_width , 'int32')

        

#        areas = [[(

#                    i*h_step, 

#                    j*w_step, 

#                    (i+1)*h_step if i+1 < pooled_height else region_height, 

#                    (j+1)*w_step if j+1 < pooled_width else region_width

#                   ) 

#                   for j in range(pooled_width)] 

#                  for i in range(pooled_height)]

        

#        # take the maximum of each area and stack the result

#        def pool_area(x): 

#          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

        

#        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

#        return pooled_features


img = np.zeros((640, 960, 3)) #Placeholder for img file h*w*c
img_w = 960
img_h = 640
scale= [128, 256, 512]
ratio= [0.5, 1, 2]

sr = len(scale) * len(ratio)
anchor_boxes = data_gen.generate_anchor_boxes(img_w, img_h, stride=16, scale=scale, ratio = ratio, no_exceed_bound = True)

#Need to create ytrain based on how img is processed

#Need to redefine loss functino to fit shape [bs, 21600, 6]
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

def get_rp(ypred, anchor_boxes, scale, ratio, score_thresh = 0.7):
    #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
    #but the way we store anchorboxes is by width first, then height
    proposed_region = []
    ypred_class = ypred[:,:2]
    ypred_regr = ypred[:,2:]
    for i in range(ypred.shape[0]):
        if ypred_class[i,0] > 0.7:
            q = i // 60*9
            sr = i % 9
            p = (i // sr) % 60
            box = anchor_boxes[p,q,sr]
            xa = box[0]
            ya = box[1]
            wa = box[2]
            ha = box[3]
            dx = ypred_regr[i,0]
            dy = ypred_regr[i,1]
            dw = ypred_regr[i,2]
            dh = ypred_regr[i,3]
            x = dx * wa + xa
            y = dy * ha + ya
            w = wa * np.exp(dw)
            h = ha * np.exp(dh)
            score = ypred_class[i,0]
            proposed_region.append([x,y,w,h,score])
    return proposed_region

input_shape = img.shape
#Transfers VGG16 layer until last convolutional layer
model = tf.keras.applications.vgg16.VGG16(input_shape = (640,960,3),weights="imagenet", include_top=False)
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

model_img_f = "D:\\Programming\\Python\\DSTA CV\\model.png"
keras.utils.plot_model(model, to_file=model_img_f, show_shapes=True)

#APPLY get_rp then nms on the output of rpn
#Feed that into a roi pooling layer