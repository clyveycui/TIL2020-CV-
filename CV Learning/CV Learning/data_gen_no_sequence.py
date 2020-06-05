import tensorflow as tf
import numpy as np
from PIL import Image
from math import log
import sys

def iou(anchor_box, ground_truth_box, img_w, img_h):
    #anchor_box, ground_truth_box are size 4 arrays in the form [x_centre, y_centre, width, height]
    #img_w, img_h are the pixel dimensions of the image

    #returns a float representing IoU value of the two boxes

    xa_min = max(anchor_box[0]-0.5*anchor_box[2], 0)
    ya_min = max(anchor_box[1]-0.5*anchor_box[3], 0)
    xa_max = min(anchor_box[0]+0.5*anchor_box[2], img_w)
    ya_max = min(anchor_box[1]+0.5*anchor_box[3], img_h)
    xg_min = max(ground_truth_box[0]-0.5*ground_truth_box[2], 0)
    yg_min = max(ground_truth_box[1]-0.5*ground_truth_box[3], 0)
    xg_max = min(ground_truth_box[0]+0.5*ground_truth_box[2], img_w)
    yg_max = min(ground_truth_box[1]+0.5*ground_truth_box[3], img_h)

    i_xmin = max(xa_min,xg_min)
    i_ymin = max(ya_min,yg_min)
    i_xmax = min(xa_max,xg_max)
    i_ymax = min(ya_max,yg_max)

    if (i_xmin>=i_xmax or i_ymin>=i_ymax):
        return 0

    i_area = (i_xmax-i_xmin)*(i_ymax-i_ymin)
    u_area = (xa_max-xa_min)*(ya_max-ya_min) + (xg_max-xg_min)*(yg_max-yg_min) - i_area
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

def preprocess_anchor_boxes(anchor_boxes):
    #takes in an output from generate_anchor_boxes

    #returns a n*4 array containing x,y,w,h of anchor boxes. Order is anchor box at 0,0 0,1 0,2 ... 0,q, 1,0, 1,1 ... p,q
    arr_shape = anchor_boxes.shape
    arr_flat = anchor_boxes.flatten()
    temp_arr = []
    arr_res = []
    for v in range(len(arr_flat)//4):
        temp_arr = [arr_flat[4*(v)], arr_flat[4*(v)+1], arr_flat[4*(v)+2], arr_flat[4*(v)+3]]
        arr_res.append(temp_arr)
    arr_res_np = np.array(arr_res)

    return arr_res_np, arr_shape
        
    
def unflatten_anchor_boxes_arr(flat_arr, arr_shape):
    #Inverse of preprocess_anchor_boxes
    #array shape should be [p, q, sr, 4]

    #returns a p*q*(s*r)*4 array

    arr_res = np.zeros(arr_shape)
    arr_len = len(flat_arr)

    for i,arr in enumerate(flat_arr):
        sr = i%arr_shape[2]
        q = (i//arr_shape[2])%arr_shape[1]
        p = i//(arr_shape[2]*arr_shape[1])
        arr_res[p,q,sr] = arr

    return arr_res

def get_unflatten_index(flat_index, arr_shape):
    p = flat_index//(arr_shape[2]*arr_shape[1])
    q = (flat_index//arr_shape[2])%arr_shape[1]
    sr = flat_index%arr_shape[2]
    return p, q ,sr

def xywh_to_xyxy(box):
    #only used for NMS pruning
    x, y, w, h = box
    x_min = x-w/2.0
    x_max = x+w/2.0
    y_min = y-h/2.0
    y_max = y+h/2.0
    return (x_min,y_min,x_max,y_max)

# Malisiewicz et al.
def non_max_suppression_fast(boxes, iou, overlapThresh):
    #boxes is an array of shape n*4 where n is the number of boxes and each box is in the form of (x1,y1,x2,y2)
    #iou is the score of all the boxes against a gt box

    #returns a 1D array containing all the indices of the boxes chosen

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

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
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        overlap_box_indices = idxs[np.where(overlap > overlapThresh)[0]]
        if len(overlap_box_indices) > 0:
            i = overlap_box_indices[np.argmax(iou[overlap_box_indices])]
        pick.append(i)
        #iou_scores
        

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
	        np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
    return pick

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
        pos_res = np.random.choice(pos_res, size=128, replace = False)
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
    anchor_boxes = generate_anchor_boxes(img_arr.shape[1], img_arr.shape[0], stride=16, scale=[32,64,128], ratio=[0.5,1,2], no_exceed_bound = True)
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
                iou_score[i,j] = iou(anchor, gt_box,img_arr.shape[1],img_arr.shape[0])
    pos_indices, neg_indices = prune_a_box(anchor_flat, iou_score)
    y_class_true = np.empty((anchorbox_arr_shape[0],anchorbox_arr_shape[1],anchorbox_arr_shape[2]*2))
    y_class_true.fill(-1)
    y_regr_true = np.zeros((anchorbox_arr_shape[0],anchorbox_arr_shape[1],anchorbox_arr_shape[2]*4))
    for ai_gti in pos_indices:
        ai = ai_gti[0]
        gti = ai_gti[1]
        p,q,rs = get_unflatten_index(ai, anchorbox_arr_shape)
        a_box = anchor_flat[ai]
        gt_box = gt_box_arr[gti]
        y_class_true[p,q,rs*2] = 1
        y_class_true[p,q,rs*2+1] = 0
        dx = (gt_box[0]-a_box[0])/a_box[2]
        dy = (gt_box[1]-a_box[1])/a_box[3]
        dw = log(gt_box[2]/a_box[2])
        dh = log(gt_box[3]/gt_box[3])
        y_regr_true[p,q,rs*4] = dx
        y_regr_true[p,q,rs*4+1] = dy
        y_regr_true[p,q,rs*4+2] = dw
        y_regr_true[p,q,rs*4+3] = dh
    for ai in neg_indices:
        p,q,rs = get_unflatten_index(ai, anchorbox_arr_shape)
        y_class_true[p,q,rs*2] = 0
        y_class_true[p,q,rs*2+1] = 1

    return y_class_true, y_regr_true
            
def preprocess_npimg(x):
        return x * 1./255.

#img = Image.open("D:\\Random pics\\joker.png")
#img_arr = np.array(img)
#labels = np.array([[1,150,84,100,100]])
#np.set_printoptions(threshold=sys.maxsize)

#y_c_true, y_r_true = create_ytrue_train(img, labels)
#print(y_c_true.shape)
#print(y_r_true.shape)
#ytrue = np.concatenate([y_c_true,y_r_true], -1)
#ytrue_class = ytrue[:,:,:18]
#ytrue_regr = np.array(ytrue[:,:,18:])
#ypred_regr = np.array(ytrue[:,:,18:])
#x = tf.where(ytrue[:,:,:18] != -1, ytrue[:,:,:18], -1)
#for i in range(ytrue_regr.shape[0]):
#    for j in range(ytrue_regr.shape[1]):
#        for n in range(ytrue_regr.shape[2]//4):
#            if ytrue_class[i,j,2*n] != 1:
#                ypred_regr[i,j,4*n] = 0
#                ypred_regr[i,j,4*n+1] = 0
#                ypred_regr[i,j,4*n+2] = 0
#                ypred_regr[i,j,4*n+3] = 0
#ypred = tf.where(ytrue[:,:,:,:18] !=-1, ypred, -1)