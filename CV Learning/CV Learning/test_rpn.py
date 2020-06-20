import os
import json
import pickle
import sys
from time import sleep
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from data_gen import create_ytrue_train, generate_anchor_boxes, preprocess_anchor_boxes
import data_gen
import tensorflow as tf
from tensorflow import keras
from roi_pooling import ROIPoolingLayer
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

def masked_regr_loss(ytrue, ypred):
    mask = tf.where(ytrue==0, 0., 1.)
    ypred = tf.multiply(mask,ypred)
    return keras.losses.Huber()(ytrue,ypred)

def get_rps(ypred,anchor_boxes):
    #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
    #but the way we store anchorboxes is by width first, then height

    #returns a np array containing all chosen regions in the form of [x,y,w,h,score]
    proposed_region = []
    res = []
    max_boxes = 18
    ypred_class = ypred[:,:2]
    ypred_regr = ypred[:,2:]
    sorted_by_score = tf.argsort(ypred_class[:,0], direction='DESCENDING')
    for i in sorted_by_score[:2000]:
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
    return np.array(proposed_region)

#json_file = r'D:\Til data\val.json'
#images_pickle = r'D:\Til data\images_train.p'
rpn_model_path = r"D:\Til data\rpn\rpn_finetuned_best.h5"
rpn = tf.keras.models.load_model(rpn_model_path, custom_objects={'rpn_regr_loss':rpn_regr_loss, 'rpn_class_loss':rpn_class_loss})
rcnn_model_path = r"D:\Til data\rcnn\best_rcnn_finetuned.h5"
rcnn = tf.keras.models.load_model(rcnn_model_path, custom_objects={'ROIPoolingLayer':ROIPoolingLayer, 'masked_regr_loss':masked_regr_loss})
anchor_boxes = generate_anchor_boxes(480,480,16,scale=[64,128,256])

#img_id = 136
#image_path = r"D:\Til data\val\val\{}.jpg".format(img_id)

#img = Image.open(image_path)
#image_size = img.size
#image_arr = np.array(img.resize((480,480)))
#if (image_arr.shape == (480,480,4)):
#    image_arr = image_arr[:,:,:3]

#img_dict ={img_id:image_arr}
#labels = []
#with open(json_file, 'r') as f:
#    annotations_dict = json.load(f)
#    annotations_list = annotations_dict['annotations']

#for annotation in annotations_list:
#    if annotation["image_id"] == img_id:
#        bbox = annotation["bbox"]
#        c = annotation["category_id"]
#        x1 = bbox[0]/image_size[0]
#        y1 = bbox[1]/image_size[1]
#        w = bbox[2]/image_size[0]
#        h = bbox[3]/image_size[1]
#        labels.append([c,x1,y1,w,h])


#font = ImageFont.truetype("arial.ttf", 64)
#img_draw = ImageDraw.Draw(img)
#for label in labels:
#    c = int(label[0])
#    x1 = int(label[1]*image_size[0]) 
#    y1 = int(label[2]*image_size[1])
#    x2 = int((label[3] + label[1])*image_size[0]) 
#    y2 = int((label[4] + label[2])*image_size[1])
#    shape = [x1,y1,x2,y2]
#    img_draw.rectangle(shape, outline="red")
#    img_draw.text((x1,y1),"category {}".format(c), font=font)

#yclass,yregr = rpn(np.array([image_arr]))
#ypred = np.concatenate((yclass,yregr), axis=-1)
#region_proposals = get_rps(ypred[0], anchor_boxes)[:12,:4]

#rp_xyxy = []
#for i,rp in enumerate(region_proposals):
#    #x1 = int(max((rp[0] - rp[2]/2), 0)*image_size[0])
#    #y1 = int(max((rp[1] - rp[3]/2), 0) *image_size[1])
#    #x2 = int(min((rp[0] + rp[2]/2), 1) *image_size[0])
#    #y2 = int(min((rp[1] + rp[3]/2), 1) *image_size[1])
#    #shape = [x1,y1,x2,y2]    
#    #img_draw.rectangle(shape, outline="blue")
#    #img_draw.text((x1,y1),"box {}".format(i), font=font)
#    x1 = max((rp[0] - rp[2]/2),0.)
#    y1 = max((rp[1] - rp[3]/2),0.)
#    x2 = min((rp[0] + rp[2]/2),1.)
#    y2 = min((rp[1] + rp[3]/2),1.)
#    if (x2 - x1) <0.1:
#        if x1 <=0.1:
#            x2 = x1 + 0.1
#        else:
#            x1 = x2-0.1
#    if (y2 - y1) <0.1:
#        if y1 <=0.1:
#            y2 = y1 + 0.1
#        else:
#            y1 = y2-0.1
#    shape = [x1,y1,x2,y2]
#    rp_xyxy.append(shape)

#scores, regr  = rcnn([np.array([image_arr]),np.array([rp_xyxy])])
#score_weights = [0.6,1,1,1,1,1]
#print(np.multiply(scores[0], score_weights).shape)
#classification = np.argmax(np.multiply(scores[0], score_weights), axis = -1)
#regr = regr[0]

#rpf = []

#for i,rp in enumerate(rp_xyxy):
#    if classification[i,0,0] != 0:
#        c = int(classification[i,0,0])
#        dx = regr[i,0,0,4*c]
#        dy = regr[i,0,0,4*c+1]
#        dw = regr[i,0,0,4*c+2]
#        dh = regr[i,0,0,4*c+3]
#        rpx1,rpy1,rpx2,rpy2 = rp
#        rpw = rpx2 - rpx1
#        rph = rpy2 - rpy1
#        xf = dx*rpw + rpx1
#        yf = dy*rph + rpy1
#        wf = rpw *np.exp(dw)
#        hf = rph *np.exp(dh)
#        x1 = int((xf)*image_size[0])
#        y1 = int((yf)*image_size[1])
#        x2 = int((xf + wf)*image_size[0])
#        y2 = int((yf + hf)*image_size[1])
#        shape = [x1,y1,x2,y2]
#        rpf.append(shape)
#        img_draw.rectangle(shape, outline="blue")
#        img_draw.text((x1,y1),"category {}".format(c), font=font)
#        img_draw.text((x2,y2),"box {}".format(i), font=font)

#img.show()

input_file = r"D:\Til data\final\CV_final_evaluation.json"
with open(input_file, "r") as f:
    test_file = json.load(f)
res = []
sample_list = [103,330, 1928, 1750]
count = 0
total = len(test_file["images"])
for info in test_file["images"]:
    img_name = info["file_name"]
    img_id = info["id"]
    img_path = os.path.join(r"D:\Til data\final\CV_final_images\CV_final_images", img_name)
    image = Image.open(img_path)
    image_size = image.size
    image_arr = np.array(image.resize((480,480)))
    if (image_arr.shape == (480,480,4)):
        image_arr = image_arr[:,:,:3]
    if (image_arr.shape == (480,480)):
        image_arr = np.stack([image_arr,image_arr,image_arr],axis = -1)
    yclass,yregr = rpn(np.array([image_arr]))
    ypred = np.concatenate((yclass,yregr), axis=-1)
    region_proposals = get_rps(ypred[0], anchor_boxes)[:12,:4]
    rp_xyxy = []
    for i,rp in enumerate(region_proposals):
        x1 = max((rp[0] - rp[2]/2),0.)
        y1 = max((rp[1] - rp[3]/2),0.)
        x2 = min((rp[0] + rp[2]/2),1.)
        y2 = min((rp[1] + rp[3]/2),1.)
        if (x2 - x1) <0.1:
            if x1 <=0.1:
                x2 = x1 + 0.1
            else:
                x1 = x2-0.1
        if (y2 - y1) <0.1:
            if y1 <=0.1:
                y2 = y1 + 0.1
            else:
                y1 = y2-0.1
        shape = [x1,y1,x2,y2]
        rp_xyxy.append(shape)

    scores, regr  = rcnn([np.array([image_arr]),np.array([rp_xyxy])])
    score_weights = [0.6,1,1,1,1,1]
    weighted_score = np.multiply(scores[0], score_weights)
    classification = np.argmax(weighted_score, axis = -1)
    regr = regr[0]
    if (img_id in sample_list):
        img_draw = ImageDraw.Draw(image)
    for i,rp in enumerate(rp_xyxy):
        if classification[i,0,0] != 0:
            c = int(classification[i,0,0])
            dx = regr[i,0,0,4*c]
            dy = regr[i,0,0,4*c+1]
            dw = regr[i,0,0,4*c+2]
            dh = regr[i,0,0,4*c+3]
            rpx1,rpy1,rpx2,rpy2 = rp
            rpw = rpx2 - rpx1
            rph = rpy2 - rpy1
            xf = max(int((dx*rpw + rpx1)*image_size[0]),0)
            yf = max(int((dy*rph + rpy1)*image_size[1]),0)
            wf = min(int((rpw *np.exp(dw))*image_size[0]),image_size[0]-xf)
            hf = min(int((rph *np.exp(dh))*image_size[1]),image_size[1]-yf)
            bbox = [xf,yf,wf,hf]
            if (img_id in sample_list):
                x1 = int(xf)
                y1 = int(yf)
                x2 = int(xf + wf)
                y2 = int(yf + hf)
                shape = [x1,y1,x2,y2]
                img_draw.rectangle(shape, outline="blue")

            if wf <= 0 or hf <= 0:
                continue
            if c != 4:
                score = scores[0,i,0,0,c]*0.6 + scores[0,i,0,0,0]*0.4
            else:
                score = scores[0,i,0,0,c] + scores[0,i,0,0,0]*0.4
            temp = {"image_id" : img_id, "category_id" : c, "bbox" : bbox, "score" : float(score)}
            res.append(temp)
    if (img_id in sample_list):
        image.show()
        sleep(3)
    image.close()
    count += 1
    print("{}/{}".format(count,total))
outfile = r"D:\Til data\final\CV_final_predictions.json"
with open(outfile, "w") as f:
    json.dump(res,f,indent=4)
