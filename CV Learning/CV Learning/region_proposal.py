from tensorflow.keras.layers import Layer
import tensorflow as tf

class RegionProposalLayer(Layer):

    def __init__(self, anchor_boxes, **kwargs):
        self.anchor_boxes = anchor_boxes
        super(RegionProposalLayer, self).__init__(**kwargs)





    def call(self, x):

        def curried_get_rps(x):
            return RegionProposalLayer._get_rps(x, self.anchor_boxes)

        rps = tf.map_fn(curried_get_rps, x, dtype=tf.float32)

        return rps

    @staticmethod

    def _get_rps(ypred,anchor_boxes):
        #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
        #but the way we store anchorboxes is by width first, then height

        #returns a np array containing all chosen regions in the form of [x,y,w,h,score]
        proposed_region = []
        res = []
        max_boxes = 18
        ypred_class = ypred[:,:2]
        ypred_regr = ypred[:,2:]
        sorted_by_score = tf.argsort(ypred_class[:,0], direction='DESCENDING')
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

        proposed_region = non_max_suppression_fast(np.array(proposed_region), overlapThresh = 0.5)
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

    @staticmethod

    def xywhs_to_xyxys(box):
        #only used for NMS pruning
        x, y, w, h, score = box
        x_min = x-w/2.0
        x_max = x+w/2.0
        y_min = y-h/2.0
        y_max = y+h/2.0
        return np.array([x_min,y_min,x_max,y_max,score])

    @staticmethod

    def xyxys_to_xywhs(box):
        #only used for NMS pruning
        x1, y1, x2, y2, score = box
        x = (x1+x2)/2.0
        y = (y1+y2)/2.0
        w = np.absolute(x1-x2)
        h = np.absolute(y1-y2)
        return np.array([x,y,w,h,score])

    @staticmethod

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

