import numpy as np
import data_gen
import json
import os
import PIL
import pickle
from PIL import Image
from math import log

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.utils.data_utils import Sequence

#We are trying to generate ytrue for rpn using a pickle file containing a dictionary of images with image id as key and a json file containing all annotations.
image_pickle = r""
json_file = r""


def get_data(pickle_file, json_file):
    #This function just loads the pickle and json file and formats the output
    #assumes images are already resized to 480*480*3
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
    anchor_boxes = data_gen.generate_anchor_boxes(480, 480, stride=16, scale=[64,128,256], ratio=[0.5,1,2], no_exceed_bound = True)
    x, y = [], []
    for img_id, labels in data_dict.items():
        print(img_id)
        x.append( imgs_dict[img_id][0])
        #most stuff happens in create_ytrue_train
        y.append( data_gen.create_ytrue_train( imgs_dict[img_id][0], np.array(labels),anchor_boxes, iou_upper=0.7, iou_lower = 0.3 ))

    #outputs x_train and y_train
    return np.array(x), np.array(y)

train_x, train_y = get_data(image_pickle,json_file)

train_x_o = r""
train_y_o = r""
with open(train_x_o, "wb") as f:
    pickle.dump(train_x,f)
with open(train_y_o, "wb") as f:
    pickle.dump(train_y,f)