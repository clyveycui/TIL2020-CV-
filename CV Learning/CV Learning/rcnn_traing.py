import tensorflow as tf
import data_gen
import numpy as np
from roi_pooling import ROIPoolingLayer
from data_gen import generate_anchor_boxes, preprocess_anchor_boxes
from tensorflow import keras
from tensorflow.keras import layers

def rpn_loss(ytrue, ypred):
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
    max_boxes = 18
    ypred_class = ypred[:,:2]
    ypred_regr = ypred[:,2:]
    print("sorting")
    sorted_by_score = tf.argsort(ypred_class[:,0], direction='DESCENDING')
    print("finished sorting")
    for i in sorted_by_score[:1000]:
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
    print("finished appending all items")
    proposed_region = non_max_suppression_fast(np.array(proposed_region), overlapThresh = 0.5)
    print("finished nms")
    proposed_region = proposed_region[:min(proposed_region.shape[0], max_boxes)]
    if proposed_region.shape[0] < max_boxes:
        boxes_needed = max_boxes - proposed_region.shape[0]
        added_in = sorted_by_score[:boxes_needed]
        new_boxes = []
        for i in added_in:
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
            new_boxes.append([x,y,w,h,score])
        proposed_region = np.append(proposed_region, new_boxes, axis=0)
    return tf.constant(proposed_region, dtype = "float32")


def get_proposal_target(proposed_region, gt_boxes, N = 9, fg_iou_thresh = 0.5, bg_iou_upper_thresh = 0.5, bg_iou_lower_thresh = 0.1 ):
    #assumes gt_box to be of the form [[category, x,y,w,h],[category, x,y,w,h]...]
    #assumes proposed_region to be of the form [[x,y,w,h,score],[x,y,w,h,score]...]
    classes = 5
    foreground_regions = []
    background_regions = []
    selected_regions = []
    regression_targets = np.zeros((2*N, 4*classes))
    regression_mask = np.zeros((2*N, 4*classes))
    class_targets = np.zeros((2*N,classes+1))
    iou_scores = np.zeros((gt_boxes.shape[0], proposed_region.shape[0]))
    for i in range(iou_scores.shape[0]):
        for j in range(iou_scores.shape[1]):
            iou_scores[i,j] = data_gen.get_overlap(proposed_region[j][:4], gt_boxes[i][1:])
            iou_scores[i,j] = data_gen.get_overlap(proposed_region[j][:4], gt_boxes[i][1:])
    iou_score_squashed = []

    for j in range(iou_scores.shape[1]):
        x = np.argsort(iou_scores[:,j])[-1]
        iou_score_squashed.append([iou_scores[x,j], x])
    iou_score_squashed = np.array(iou_score_squashed)
    print(iou_score_squashed)
    #Now, iou_score_squashed is a n*2 list containing the max iou score and the index of the gtbox that gives the max iou score for each region
    sorted_index_by_score = np.argsort(iou_score_squashed[:,0], axis=0)
    print(sorted_index_by_score)
    for i in range(proposed_region.shape[0]):
        fg_index = sorted_index_by_score[-i-1]
        bg_index = sorted_index_by_score[i]
        print(iou_score_squashed[bg_index][0])
        if iou_score_squashed[fg_index][0] >= fg_iou_thresh:
            foreground_regions.append([fg_index, int(iou_score_squashed[fg_index][1])])
        if iou_score_squashed[bg_index][0] >= bg_iou_lower_thresh and iou_score_squashed[bg_index][0] < bg_iou_upper_thresh:
            background_regions.append(bg_index)
    if len(foreground_regions) > N:
        foreground_regions = foreground_regions[:N]
    if len(background_regions) > N:
        background_regions = background_regions[:N]
    while len(foreground_regions) < N:
        foreground_regions.append(foreground_regions[np.random.choice(range(len(foreground_regions)))])
    while len(background_regions) < N:
        background_regions.append(np.random.choice(background_regions))
    for i in range(N):
        rp = proposed_region[foreground_regions[i][0]]
        gt = gt_boxes[foreground_regions[i][1]]
        gtc, gtx, gty, gtw, gth = gt
        gtc = int(gtc)
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


rpn_model_path = r"D:\Til data\rpn_model_finetuned.h5"
rpn = tf.keras.models.load_model(rpn_model_path, custom_objects={'custom_loss':rpn_loss})
rpn.trainable = False

input_shape = (640, 960, 3)
anchor_boxes = generate_anchor_boxes(960,640,16,scale=[64,128,256])
proposed_regions = None


model = tf.keras.applications.vgg16.VGG16(input_shape = (640,960,3),weights="imagenet", include_top=False)
model.layers.pop()
model_2 = tf.keras.models.Sequential(name="VGG16")
for layer in model.layers[:-1]:
    model_2.add(layer)
for layer in model_2.layers:
    layer.trainable = False

inputs = keras.Input(shape = input_shape)
fm = model_2(inputs)
#rpn_o = rpn(inputs)
#rp = get_rp(rpn_o, anchor_boxes, max_boxes = 18)
rp = keras.Input(shape =(18,4))
#Need to convert rp into a tensor of shape (batch_size, n_rois, 4) in the form of (xmin, ymin, xmax, ymax)
x = ROIPoolingLayer(7,7)([fm, rp])
x = tf.keras.layers.Dense(4096)(x)
x = tf.keras.layers.Dense(4096)(x)
object_classification = tf.keras.layers.Dense(6)(x)
object_classification = layers.Softmax(axis=-1, name="classification")(object_classification)
bbox_regr = tf.keras.layers.Dense(24, name="regression")(x)

rcnn = tf.keras.Model(inputs = [inputs,rp], outputs=[object_classification, bbox_regr], name="rcnn")
rcnn = rpn.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
            loss={"classification":tf.keras.losses.CategoricalCrossentropy(), "regression":tf.keras.losses.Huber()},
            metrics=["Accuracy",tf.keras.metrics.RootMeanSquaredError()])
rcnn.summary()

