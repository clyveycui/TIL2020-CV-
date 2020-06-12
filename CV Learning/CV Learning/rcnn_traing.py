import tensorflow as tf
import rpn_training
import data_gen
import numpy as np

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

def get_rp(ypred, anchor_boxes, max_boxes = -1, score_thresh = 0.7):
    #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
    #but the way we store anchorboxes is by width first, then height

    #returns a np array containing all chosen regions in the form of [x,y,w,h,score]
    proposed_region = []
    ypred_class = ypred[:,:2]
    ypred_regr = ypred[:,2:]
    for i in range(ypred.shape[0]):
        if ypred_class[i,0] > 0.7:
            q = i // (60*9)
            sr = i % 9
            p = (i // 9) % 60
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
    if max_boxes > 0:
        return proposed_region[:max_boxes]
    else:
        return proposed_region

def get_proposal_target(proposed_region, gt_boxes, N = 16, fg_iou_thresh = 0.5, bg_iou_upper_thresh = 0.5, bg_iou_lower_thresh = 0.1 ):
    #assumes gt_box to be of the form [[category, x,y,w,h],[category, x,y,w,h]...]
    #assumes proposed_region to be of the form [[x,y,w,h,score],[x,y,w,h,score]...]
    classes = 5
    foreground_regions = []
    background_regions = []
    selected_regions = []
    regression_targets = np.zeros((2*N, 4*classes))
    regression_mask = np.zeros((2*N, 4*classes))
    class_targets = np.zeros((2*N,classes+1))
    iou_scores = np.zeros((gt_boxes.shape[0], proposed_region.shape))
    for i in range(iou_scores.shape[0]):
        for j in range(iou_scores.shape[1]):
            iou_scores[i,j] = data_gen.iou(proposed_region[j][:4], gt_boxes[i][1:], img_w = 1.0, img_h = 1.0)
    iou_score_squashed = []
    for j in range(iou_scores.shape[1]):
        x = np.argsort(iou_scores[:,j])[-1]
        iou_score_squashed.append([iou_scores[x], x])
    #Now, iou_score_squashed is a n*2 list containing the max iou score and the index of the gtbox that gives the max iou score for each region
    sorted_index_by_score = np.argsort(iou_score_squashed, axis=0)[0]
    for i in range(min(N,proposed_region.shape[0]//2)):
        fg_index = sorted_index_by_score[-i-1]
        bg_index = sorted_index_by_score[i]
        if iou_score_squashed[fg_index][0] >= fg_iou_thresh:
            foreground_regions.append([fg_index, iou_score_squashed[fg_index][1]])
        if iou_score_squashed[bg_index][0] >= bg_iou_lower_thresh and iou_score_squashed[bg_index][0] < bg_iou_upper_thresh:
            background_regions.append(bg_index)
    while len(foreground_regions) < N:
        foreground_regions.append(foreground_regions[np.random.choice(range(len(foreground_regions)))])
    while len(background_regions) < N:
        background_regions.append(np.random.choice(background_regions))
    for i in range(N):
        rp = proposed_region[foreground_regions[i][0]]
        gt = gt_boxes[foreground_region[i][1]]
        gtc, gtx, gty, gtw, gth = gt
        rx, ry, rw, rh, _ = rp
        dx = (gtx - rx)*1.0/rw
        dy = (gty - ry)*1.0/rh
        dw = np.log(gtw/rw)
        dh = np.log(gth/rh)
        regression_targets[i,4*(gtc-1)] = dx
        regression_targets[i,4*(gtc-1)+1] = dy
        regression_targets[i,4*(gtc-1)+2] = dw
        regression_targets[i,4*(gtc-1)+3] = dh
        regression_mask[i,4*(gtc-1)] = 1
        regression_mask[i,4*(gtc-1)+1] = 1
        regression_mask[i,4*(gtc-1)+2] = 1
        regression_mask[i,4*(gtc-1)+3] = 1
        class_targets[i,gtc] = 1
        selected_regions.append(rp[:4])
    for i in range(N):
        class_targets[N+i, 0] = 1
        selected_regions.append(proposed_region[background_regions[N-i-1]][:4])

    return np.array(selected_regions), regression_targets, regression_mask, class_targets 


#Source : https://gist.github.com/Jsevillamol/0daac5a6001843942f91f2a3daea27a7

model_path = ""
rpn = tf.keras.models.load_model(load_model_path, custom_objects={'custom_loss':rpn_training.custom_loss})

model = tf.keras.applications.vgg16.VGG16(input_shape = (640,960,3),weights="imagenet", include_top=False)
model.layers.pop()
model_2 = tf.keras.models.Sequential(name="VGG16")
for layer in model.layers[:-1]:
    model_2.add(layer)
for layer in model_2.layers:
    layer.trainable = False