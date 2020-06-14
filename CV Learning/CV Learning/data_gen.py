import img_aug
import numpy as np
import json
import os
import PIL
from PIL import Image
from math import log
from multiprocessing import Pool

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.utils.data_utils import Sequence

#THIS MODULE ONLY DEFINES THE SEQUENCE OBJECT FOR RPN 
#NEED TO MODIFY IT FOR FAST RCNN TRAINING
#Try redoing using different sizes for the rpn


def iou(anchor_box, ground_truth_box):
    #anchor_box, ground_truth_box are size 4 arrays in the form [x_centre, y_centre, width, height]
    #img_w, img_h are the pixel dimensions of the image

    #returns a float representing IoU value of the two boxes

    xa_min = max(anchor_box[0]-0.5*anchor_box[2], 0)
    ya_min = max(anchor_box[1]-0.5*anchor_box[3], 0)
    xa_max = min(anchor_box[0]+0.5*anchor_box[2], 1)
    ya_max = min(anchor_box[1]+0.5*anchor_box[3], 1)
    xg_min = max(ground_truth_box[0]-0.5*ground_truth_box[2], 0)
    yg_min = max(ground_truth_box[1]-0.5*ground_truth_box[3], 0)
    xg_max = min(ground_truth_box[0]+0.5*ground_truth_box[2], 1)
    yg_max = min(ground_truth_box[1]+0.5*ground_truth_box[3], 1)

    i_xmin = max(xa_min,xg_min)
    i_ymin = max(ya_min,yg_min)
    i_xmax = min(xa_max,xg_max)
    i_ymax = min(ya_max,yg_max)

    if (i_xmin>=i_xmax or i_ymin>=i_ymax):
        return 0

    i_area = (i_xmax-i_xmin)*(i_ymax-i_ymin)
    u_area = (xa_max-xa_min)*(ya_max-ya_min) + (xg_max-xg_min)*(yg_max-yg_min) - i_area
    return (i_area/u_area)

def get_overlap(anchor_box, ground_truth_box):
    #anchor_box, ground_truth_box are size 4 arrays in the form [x_centre, y_centre, width, height]
    #img_w, img_h are the pixel dimensions of the image

    #returns a float representing the percentage of the anchor box overlapping with the gt box

    xa_min = max(anchor_box[0]-0.5*anchor_box[2], 0)
    ya_min = max(anchor_box[1]-0.5*anchor_box[3], 0)
    xa_max = min(anchor_box[0]+0.5*anchor_box[2], 1)
    ya_max = min(anchor_box[1]+0.5*anchor_box[3], 1)
    xg_min = max(ground_truth_box[0]-0.5*ground_truth_box[2], 0)
    yg_min = max(ground_truth_box[1]-0.5*ground_truth_box[3], 0)
    xg_max = min(ground_truth_box[0]+0.5*ground_truth_box[2], 1)
    yg_max = min(ground_truth_box[1]+0.5*ground_truth_box[3], 1)

    i_xmin = max(xa_min,xg_min)
    i_ymin = max(ya_min,yg_min)
    i_xmax = min(xa_max,xg_max)
    i_ymax = min(ya_max,yg_max)

    if (i_xmin>=i_xmax or i_ymin>=i_ymax):
        return 0

    i_area = (i_xmax-i_xmin)*(i_ymax-i_ymin)
    u_area = (xa_max-xa_min)*(ya_max-ya_min)
    return (i_area/u_area)



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
                a_box = np.array([anchor_points[i,j][0]*1.0/w, anchor_points[i,j][1]*1.0/h, sl[0]/w, sl[1]/h])
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

def preprocess_anchor_boxes(anchor_boxes):
    #takes in an output from generate_anchor_boxes

    #returns a n*4 array containing x,y,w,h of anchor boxes. Order is anchor box at 0,0 0,1 0,2 ... 0,q, 1,0, 1,1 ... p,q
    arr_shape = anchor_boxes.shape
    arr_res =[]
    for q in range(arr_shape[1]):
        for p in range(arr_shape[0]):
            for sr in range(arr_shape[2]):
                arr_res.append(anchor_boxes[p,q,sr])
    return np.array(arr_res), arr_shape
    #arr_flat = anchor_boxes.flatten()
    #temp_arr = []
    #arr_res = []
    #for v in range(len(arr_flat)//4):
    #    temp_arr = [arr_flat[4*(v)], arr_flat[4*(v)+1], arr_flat[4*(v)+2], arr_flat[4*(v)+3]]
    #    arr_res.append(temp_arr)
    #arr_res_np = np.array(arr_res)

    #return arr_res_np, arr_shape
       


def iou_sampling(pruned_anchor_box_indices, iou_scores, iou_upper=0.7, iou_lower=0.3):
    #pruned_anchor_box_indices is the index of all anchor boxes
    #iou_scores is the unpruned iou_scores
    #iou_upper, iou_lower are floats between 1 and 0 representing the iou threshold for positive cases and negative cases

    #returns 2 arrays containing 256 entries.
    #pos_res is in the shape p*2, where each entry is in the form of (i, j) where i is the index of the anchor box chosen and j is the corresponding bounding box
    #neg_res is in the shape n where each entry is the index of the anchor box chosen

    #This part gets the positives
    pos_res = []
    pos_case_indices = []
    candidate_indices = pruned_anchor_box_indices
    indices_to_del = []
    for i,_ in enumerate(iou_scores):
        idx = np.argmax(iou_scores[i])
        if iou_scores[i,idx] != 0:
          pos_res.append([idx,i])
        pos_case_indices.append(idx)
        candidate_indices = candidate_indices[candidate_indices != idx]
        for f,j in enumerate(candidate_indices):
            if iou_scores[i,j] > iou_upper:
                pos_res.append([j,i])
                indices_to_del.append(f)
                pos_case_indices.append(j)
        candidate_indices = np.delete(candidate_indices, indices_to_del)
        indices_to_del = []

    pos_res = np.array(pos_res)
    if pos_res.shape[0] > 128:
        pos_res = pos_res[np.random.choice([i for i in range(pos_res.shape[0])], size=128, replace = False)]
    pos_count = pos_res.shape[0]

    #This part gets the negatives
    negative_candidate_indices = [i for i in range(iou_scores.shape[1]) if i not in pos_case_indices ]
    neg_candidates = [i for i in negative_candidate_indices if np.all(iou_scores[:,i]<iou_lower) and np.all(iou_scores[:,i]>=0)]
    neg_res = []
    if len(neg_candidates) <= 256-pos_count:
        neg_res = neg_candidates
    else:
        neg_res = np.random.choice(neg_candidates, size=256-pos_count, replace = False)
    return pos_res, neg_res

def prune_a_box(anchor_boxes_flat, iou_scores):
    anchor_box_indices = np.array(range(anchor_boxes_flat.shape[0]))
    pos_box_indices, neg_box_indices = iou_sampling(anchor_box_indices, iou_scores, iou_upper=0.7, iou_lower=0.3)
    return pos_box_indices, neg_box_indices



def create_ytrue_train(img, labels, iou_upper = 0.7, iou_lower = 0.3):
    img_arr = np.array(img)
    anchor_boxes = generate_anchor_boxes(img_arr.shape[1], img_arr.shape[0], stride=16, scale=[64,128,256], ratio=[0.5,1,2], no_exceed_bound = True)
    gt_box_arr = []
    for label in labels:
        gtclass, gtx, gty, gtw, gth = label
        gtclass = int(gtclass)
        #Might change depends on how label is given
        gt_bbox = [gtx, gty, gtw, gth]
        gt_box_arr.append(gt_bbox)

    anchor_flat, anchorbox_arr_shape = preprocess_anchor_boxes(anchor_boxes)
    iou_score = np.zeros((len(gt_box_arr), len(anchor_flat)))
    for i,gt_box in enumerate(gt_box_arr):
        for j,anchor in enumerate(anchor_flat):
            if (anchor[2] == 0):
                iou_score[i,j] = -1
            else:
                iou_score[i,j] = iou(anchor, gt_box)
    pos_indices, neg_indices = prune_a_box(anchor_flat, iou_score)
    y_class_true = np.zeros((anchor_flat.shape[0], 2))
    y_regr_true = np.zeros((anchor_flat.shape[0], 4))
    for ai_gti in pos_indices:
        ai = ai_gti[0]
        gti = ai_gti[1]
        a_box = anchor_flat[ai]
        gt_box = gt_box_arr[gti]
        y_class_true[ai, 0] = 1
        y_class_true[ai, 1] = 0
        dx = (gt_box[0]-a_box[0])/a_box[2]
        dy = (gt_box[1]-a_box[1])/a_box[3]
        dw = log(gt_box[2]/a_box[2])
        dh = log(gt_box[3]/a_box[3])

        y_regr_true[ai,0] = dx
        y_regr_true[ai,1] = dy
        y_regr_true[ai,2] = dw
        y_regr_true[ai,3] = dh
    for ai in neg_indices:
        y_class_true[ai,0] = 0
        y_class_true[ai,1] = 1

    return np.concatenate([y_class_true, y_regr_true], -1)

class TILSequence(Sequence):
    def __init__(self, pickle_file, json_annotation_file, batch_size, augment_fn, testmode=False):
        self._prepare_data(pickle_file, json_annotation_file)
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.input_wh = (960,640,3)
        self.testmode = testmode
    
    def _prepare_data(self, pickle_file, json_annotation_file):
        with open(pickle_file, 'rb') as f:
            imgs_dict = pickle.load(f)
        data_dict = {}
        for imgid in imgs_dict:
            data_dict[imgid] = []
        with open(json_annotation_file, 'r') as f:
            annotations_dict = json.load(f)
        #annotations_list is a list containing all annotations in the form of dictionaries     {
        #  "area": area_of_bounding_box,
        #  "iscrowd": ?,
        #  "id": id_of_annotation,
        #  "image_id": id_of_corresponding_image,
        #  "category_id": category_id_of_object,
        #  "bbox": [x,y,w,h]
        #}
        annotations_list = annotations_dict['annotations']
        for annotation in annotations_list:
            try:
                img_id = str(annotation['image_id'])
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
        #anchor_boxes = generate_anchor_boxes(960,640, 16, scale = [128,256,512], ratio = [0.5,1.0,2.0], no_exceed_bound = True)

        self.x, self.y, self.ids = [], [], []
        for img_id, labels in data_dict.items():
            self.x.append( imgs_dict[img_id][0])
            self.y.append( np.array(labels) )
            self.ids.append( img_id )

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
  
    def __getitem__(self, idx):
        #return self.get_batch_test(idx) if self.testmode else self.get_batch(idx)
        return self.get_batch(idx)

    def preprocess_fn(self, x):
        return x/255.0


    #def get_batch_test(self, idx):
    #    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    #    batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]

    #    x_acc, y_acc = [], {}
    #    original_img_dims = []
    #    with Pool(self.batch_size) as p:
    #        # Read in the PIL objects from filepaths
    #        batch_x = p.map(load_img, batch_x)
    #    for i, image in enumerate(batch_x):
    #        batch_x[i] = image.resize((960,640))
    #    for x,y in zip( batch_x, batch_y ):
    #        W,H = x.size
    #        original_img_dims.append( (W,H) )
    #        x_aug, y_aug = self.augment_fn( x, y )
    #        if x_aug.size != self.input_wh[:2]:
    #            x_aug.resize( self.input_wh )
    #        x_acc.append( np.array(x_aug) )
    #        y_dict = self.label_encoder( y_aug )
    #        for dimkey, label in y_dict.items():
    #            if dimkey not in y_acc:
    #                y_acc[dimkey] = []
    #            y_acc[dimkey].append( label )

    #    return batch_ids, original_img_dims, self.preprocess_fn( np.array( x_acc ) ), { dimkey: np.array( gt_tensor ) for dimkey, gt_tensor in y_acc.items() }

    def get_batch(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x_acc, y_acc = [], []
    
        for x,y in zip( batch_x, batch_y ):
            x_aug, y_aug = self.augment_fn( Image.fromarray(x, 'RGB'), y )
            if x_aug.size != (960,640):
                x_aug = x_aug.resize( (960,640) )
            x_acc.append( np.array(x_aug) )
            ytrue = create_ytrue_train(x_aug, y_aug, iou_upper = 0.7, iou_lower = 0.3)
            y_acc.append( ytrue )

        return self.preprocess_fn( np.array( x_acc ) ), np.array( y_acc )