import numpy as np
import data_gen
import json
import os
import PIL
import pickle
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

def xywhs_to_xyxys(box):
    #only used for NMS pruning
    x, y, w, h, score = box
    x_min = x-w/2.0
    x_max = x+w/2.0
    y_min = y-h/2.0
    y_max = y+h/2.0
    return np.array([x_min,y_min,x_max,y_max,score])

def xyxys_to_xywhs(box):
    #only used for NMS pruning
    x1, y1, x2, y2, score = box
    x = (x1+x2)/2.0
    y = (y1+y2)/2.0
    w = np.absolute(x1-x2)
    h = np.absolute(y1-y2)
    return np.array([x,y,w,h,score])

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    #boxes is an array of shape n*5 where n is the number of boxes and each box is in the form of (x,y,w,h,s)

    #returns a p*5 array containing all the boxes of the boxes picked in the form of (x,y,w,h,s)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    #converts from xywhs to xyxys
    for i in range(boxes.shape[0]):
        boxes[i] = xywhs_to_xyxys(boxes[i])

    #sorts the array based on score
    scores = boxes[:,4]
    sorted_index = np.argsort(scores)
    boxes_sorted = []
    for i in sorted_index:
        boxes_sorted.append(boxes[i])
    boxes_sorted = np.array(boxes_sorted)

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes_sorted[:,0]
    y1 = boxes_sorted[:,1]
    x2 = boxes_sorted[:,2]
    y2 = boxes_sorted[:,3]

    area = (x2 - x1 ) * (y2 - y1)
    idxs = np.array([i for i in range(len(y2))])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]


        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        overlap_box_indices = idxs[np.where(overlap > overlapThresh)[0]]
        pick.append(i)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
	        np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
    picked_boxes = []
    for p in pick:
        picked_boxes.append(boxes_sorted[p])
    #converts back to xywhs form
    for i in range(np.array(picked_boxes).shape[0]):
        picked_boxes[i] = xyxys_to_xywhs(picked_boxes[i])
    return np.array(picked_boxes)




def get_rps(ypred,anchor_boxes):
    #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
    #but the way we store anchorboxes is by width first, then height

    #returns a np array containing all chosen regions in the form of [x,y,w,h,score]
    proposed_region = []
    res = []
    ypred_class = ypred[:,:2]
    ypred_regr = ypred[:,2:]
    sorted_by_score = tf.argsort(ypred_class[:,0], direction='DESCENDING')
    for i in sorted_by_score[:1000]:
        q = i // (30*9)
        sr = i % 9
        p = (i // 9) % 30
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
    proposed_region = non_max_suppression_fast(np.array(proposed_region), overlapThresh = 0.5)
    return np.array(proposed_region, dtype = "float32")

def get_overlap(rp, gt_box):
    _,x,y,w,h = gt_box
    x1,y1,x2,y2 = rp
    xg1 = x
    yg1 = y
    xg2 = x + w
    yg2 = y + h
    rpa = (x2-x1)*(y2-y1)
    xi1 =max(xg1,x1)
    yi1 = max(yg1, y1)
    xi2 = min(xg2, x2)
    yi2 = min(yg2, y2)
    return max((xi2-xi1),0.0) * max((yi2-yi1),0.0)/rpa

def get_training_data_rcnn(region_proposals, gt_boxes):
    #assumes region proposals are in the form of [x,y,w,h,s], gt_boxes are in the form of [c,x_top,y_left,w,h]
    fg_thresh = 0.4
    bg_upper_thresh = 0.4
    bg_lower_thresh = 0.1
    classes = 6
    regr_target = np.zeros((12,1,1,24))
    class_target = np.zeros((12,1,1,6))
    roi_in = []
    checked_rps = []
    overlap_score = np.zeros((len(gt_boxes), len(region_proposals)))
    total_boxes =12
    pos_indices = []
    neg_indices = []
    for rp in region_proposals:
        x1 = max(rp[0] - rp[2]/2.0, 0.)
        y1 =  max(rp[1] - rp[3]/2.0, 0.)
        x2 = min(rp[0] + rp[2]/2.0, 1.)
        y2 = min(rp[1] + rp[3]/2.0, 1.)
        if (x2-x1) < 0.1:
            if x1 <=0.1:
                x2 = x1 + 0.1 
            else:
                x1 = x2 - 0.1
        if (y2-y1) < 0.1:
            if y1 <=0.1+1./480:
                y2 = y1 + 0.1 
            else:
                y1 = y2 - 0.1 
        checked_rps.append([x1,y1,x2,y2])
    for i in range(overlap_score.shape[0]):
        for j in range(overlap_score.shape[1]):
            overlap_score[i,j] = data_gen.iou_modified(checked_rps[j], gt_boxes[i])
    box_with_max_score = np.argsort(overlap_score, axis=0)[-1]
    max_score_per_region = [overlap_score[gt_i,i] for i,gt_i in enumerate(box_with_max_score)]
    sorted_by_max_score = np.argsort(max_score_per_region)

    for i in sorted_by_max_score[::-1]:
        gti = box_with_max_score[i]
        if max_score_per_region[i] >= fg_thresh:
            pos_indices.append([i,gti])
        elif max_score_per_region[i]  >= bg_lower_thresh and max_score_per_region[i]  < bg_upper_thresh :
            neg_indices.append(i)

    p = len(pos_indices)
    q = len(neg_indices)
    if p+q == 0:
        print("PREMATURE EXIT -------------------")
        return np.zeros(1), None, None
    if p >= 6 and q >= 6:
        pos_indices = pos_indices[:6]
        neg_indices = neg_indices[:6]
    elif p < 6 and q >= 12 - p:
        neg_indices = neg_indices[:12-p]
    elif q < 6 and p > 12 - q:
        pos_indices = pos_indices[:12-q]
    elif p +q < 12:
        needed = 12-(p+q)
        if(p >0):
            for i, rpi_gti in enumerate(pos_indices):
                gti = rpi_gti[1]
                c = gt_boxes[gti][0]
                if c != 4:
                    for i in range(needed):
                        pos_indices.append(rpi_gti)
                if len(pos_indices) + len(neg_indices) == 12:
                    break
            if len(pos_indices) + len(neg_indices) != 12:
                added = np.random.choice([i for i in range(p)], needed)
                for i in added:
                    pos_indices.append(pos_indices[i])
        else:
            added = np.random.choice(neg_indices, needed)
            neg_indices = np.concatenate((neg_indices,added))

    p = len(pos_indices)
    q = len(neg_indices)
    for i,rpi_gti in enumerate(pos_indices):
        rpi = rpi_gti[0]
        gti = rpi_gti[1]
        roi_in.append(checked_rps[rpi])
        c,gtx,gty,gth,gtw = gt_boxes[gti]
        class_target[i,0,0,c] = 1
        rpx,rpy,rpx2,rpy2 = checked_rps[rpi]
        rpw = rpx2 - rpx
        rph = rpy2 - rpy
        dx = (gtx - rpx) / rpw
        dy = (gty - rpy) / rph
        dw = np.log(gtw/rpw)
        dh = np.log(gth/rph)
        regr_target[i,0,0,c*4] = dx
        regr_target[i,0,0,c*4+1] = dy
        regr_target[i,0,0,c*4+2] = dw
        regr_target[i,0,0,c*4+3] = dh
    for i,rpi in enumerate(neg_indices):
        roi_in.append(checked_rps[rpi])
        class_target[p+i,0,0,0] = 1

    return np.array(roi_in), regr_target, class_target

def pickle_rcnn_training_data(image_pickle, json_file,rpn):
    with open(image_pickle, 'rb') as f:
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
            imwidth  = imgs_dict[img_id][1]
            imheight = imgs_dict[img_id][2]
            c = annotation['category_id'] # TODO: make sure that category ids start from 1, not 0. Need to set back_ground as another category
            boxleft,boxtop,boxwidth,boxheight = annotation['bbox']
            x,y,w,h = boxleft/imwidth, boxtop/imheight, boxwidth/imwidth, boxheight/imheight
            data_dict[img_id].append( [c,x,y,w,h] )
        except KeyError:
            continue
    anchor_boxes = data_gen.generate_anchor_boxes(480, 480, stride=16, scale=[64,128,256], ratio=[0.5,1,2])
    x_img = []
    x_rp = []
    y_class =[]
    y_regr = []
    count = 1
    skipped = 0
    n = len(data_dict)
    for img_id, labels in data_dict.items():
        print("{}/{}, {} skipped".format(count, n, skipped))
        x_img.append( imgs_dict[img_id][0])
        rpn_class, rpn_regr = rpn(np.array([imgs_dict[img_id][0]]))
        rpn_o = np.concatenate((rpn_class,rpn_regr), axis = -1)
        region_proposals = get_rps(rpn_o[0], anchor_boxes)
        roi_in,regr_target,class_target = get_training_data_rcnn(region_proposals, labels)
        if (np.all(roi_in == 0)):
            skipped += 1
            x_img.pop()
            continue
        x_rp.append(roi_in)
        y_class.append(class_target)
        y_regr.append(regr_target)
        count += 1
    return np.array(x_img), np.array(x_rp), np.array(y_class), np.array(y_regr)



rpn_model_path = r"D:\Til data\rpn\rpn_finetuned_best.h5"
rpn = tf.keras.models.load_model(rpn_model_path, custom_objects={'rpn_regr_loss':rpn_regr_loss, 'rpn_class_loss':rpn_class_loss})
rpn.trainable = False
image_pickle = r"D:\Til data\images.p"
json_file = r"D:\Til data\train.json"
ximg_out = r"D:\Til data\rcnn\rcnn_img_x.p"
xrp_out = r"D:\Til data\rcnn\rcnn_rp_x.p"
yclass_out = r"D:\Til data\rcnn\rcnn_class_y.p"
yregr_out = r"D:\Til data\rcnn\rcnn_regr_y.p"


x_img, x_rp, y_class, y_regr = pickle_rcnn_training_data(image_pickle, json_file,rpn)
print(x_img.shape)
print(x_rp.shape)
print(y_class.shape)
print(y_regr.shape)

count = [0,0,0,0,0,0]
for i in range(y_class.shape[0]):
    for j in range(y_class.shape[1]):
        count = np.add(count, y_class[i,j,0,0])
print(np.array(count,dtype="int32"))

with open(ximg_out, "wb") as f:
    pickle.dump(x_img, f)
with open(xrp_out, "wb") as f:
    pickle.dump(x_rp, f)
with open(yclass_out, "wb") as f:
    pickle.dump(y_class, f)
with open(yregr_out, "wb") as f:
    pickle.dump(y_regr, f)