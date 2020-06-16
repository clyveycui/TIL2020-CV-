
import json
import pickle
import sys
from PIL import Image, ImageDraw
import numpy as np
from data_gen import create_ytrue_train, generate_anchor_boxes, preprocess_anchor_boxes
import data_gen
import tensorflow as tf
from tensorflow import keras

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

def get_rps(ypred,anchor_boxes):
    #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
    #but the way we store anchorboxes is by width first, then height

    #returns a np array containing all chosen regions in the form of [x,y,w,h,score]
    proposed_region = []
    res = []
    max_boxes = 9
    ypred_class = ypred[:,:2]
    ypred_regr = ypred[:,2:]
    sorted_by_score = tf.argsort(ypred_class[:,0], direction='DESCENDING')
    for i in sorted_by_score[:2000]:
        if(ypred_class[i,0]>0.7):
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
    proposed_region = proposed_region[:min(proposed_region.shape[0], max_boxes)]
    #if proposed_region.shape[0] < max_boxes:
    #    boxes_needed = max_boxes - proposed_region.shape[0]
    #    added_in = sorted_by_score[:boxes_needed]
    #    new_boxes = []
    #    for i in added_in:
    #        q = i // (30*9)
    #        sr = i % 9
    #        p = (i // 9) % 30
    #        box = anchor_boxes[p,q,sr]
    #        xa = box[0]
    #        ya = box[1]
    #        wa = box[2]
    #        ha = box[3]
    #        dx = ypred_regr[i,0]
    #        dy = ypred_regr[i,1]
    #        dw = ypred_regr[i,2]
    #        dh = ypred_regr[i,3]
    #        x = dx * wa + xa
    #        y = dy * ha + ya
    #        w = wa * np.exp(dw)
    #        h = ha * np.exp(dh)
    #        score = ypred_class[i,0]
    #        new_boxes.append([x,y,w,h,score])
    #    proposed_region = np.append(proposed_region, new_boxes, axis=0)
    return np.array(proposed_region)

json_file = r'D:\Til data\val.json'
#images_pickle = r'D:\Til data\images_train.p'
rpn_model_path = r"D:\Til data\rpn_model_finetuned_2.h5"
rpn = tf.keras.models.load_model(rpn_model_path, custom_objects={'custom_loss':custom_loss})
rpn.trainable = False
anchor_boxes = generate_anchor_boxes(480,480,16,scale=[64,128,256])

image_path = r"D:\Til data\val\val\343.jpg"
img_id = 343
img = Image.open(image_path)
image_size = img.size
image_arr = np.array(img.resize((480,480)))
if (image_arr.shape == (480,480,4)):
    image_arr = image_arr[:,:,:3]

img_dict ={img_id:image_arr}
labels = []
with open(json_file, 'r') as f:
    annotations_dict = json.load(f)
    annotations_list = annotations_dict['annotations']

for annotation in annotations_list:
    if annotation["image_id"] == img_id:
        bbox = annotation["bbox"]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = (bbox[0] + bbox[2])
        y2 = (bbox[1] + bbox[3])
        labels.append([x1,y1,x2,y2])



img_draw = ImageDraw.Draw(img)
for label in labels:
    x1 = label[0]
    y1 = label[1]
    x2 = label[2]
    y2 = label[3]
    shape = [x1,y1,x2,y2]
    img_draw.rectangle(shape, outline="red")

ypred = rpn(np.array([image_arr]))
region_proposals = get_rps(ypred[0], anchor_boxes)
print(region_proposals)
for rp in region_proposals:
    x1 = int((rp[0] - rp[2]/2) *image_size[0])
    y1 = int((rp[1] - rp[3]/2) *image_size[1])
    x2 = int((rp[0] + rp[2]/2) *image_size[0])
    y2 = int((rp[1] + rp[3]/2) *image_size[1])
    shape = [x1,y1,x2,y2]
    print(shape)
    img_draw.rectangle(shape, outline="blue")


img.show()