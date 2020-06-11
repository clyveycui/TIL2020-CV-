import tensorflow as tf
import rpn_training
import data_gen
import numpy as np

def get_rp(ypred, anchor_boxes, scale, ratio, max_boxes = -1, score_thresh = 0.7):
    #ypred is a flattened depth first. so it is h1w1s1r1, h1w1s1r2...h1w2s1r1, h1w2s1r2 ... h2w1s1r1 ...
    #but the way we store anchorboxes is by width first, then height

    #returns a np array containing all chosen regions in the form of [x,y,w,h,score]
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

    proposed_region = data_gen.non_max_suppression_fast(proposed_region, overlapThresh = 0.5)
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

    return np.array(selected_regions), regression_targets, regregression_mask, class_targets 

model_path = ""
rpn = tf.keras.models.load_model(load_model_path, custom_objects={'custom_loss':rpn_training.custom_loss})


#Source : https://gist.github.com/Jsevillamol/0daac5a6001843942f91f2a3daea27a7
class ROIPoolingLayer(Layer):

    """ Implements Region Of Interest Max Pooling 

        for channel-first images and relative bounding box coordinates

        

        # Constructor parameters

            pooled_height, pooled_width (int) -- 

              specify height and width of layer outputs

        

        Shape of inputs

            [(batch_size, pooled_height, pooled_width, n_channels),

             (batch_size, num_rois, 4)]

           

        Shape of output

            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    

    """

    def __init__(self, pooled_height, pooled_width, **kwargs):

        self.pooled_height = pooled_height

        self.pooled_width = pooled_width

        

        super(ROIPoolingLayer, self).__init__(**kwargs)

        

    def compute_output_shape(self, input_shape):

        """ Returns the shape of the ROI Layer output

        """

        feature_map_shape, rois_shape = input_shape

        assert feature_map_shape[0] == rois_shape[0]

        batch_size = feature_map_shape[0]

        n_rois = rois_shape[1]

        n_channels = feature_map_shape[3]

        return (batch_size, n_rois, self.pooled_height, 

                self.pooled_width, n_channels)



    def call(self, x):

        """ Maps the input tensor of the ROI layer to its output

        

            # Parameters

                x[0] -- Convolutional feature map tensor,

                        shape (batch_size, pooled_height, pooled_width, n_channels)

                x[1] -- Tensor of region of interests from candidate bounding boxes,

                        shape (batch_size, num_rois, 4)

                        Each region of interest is defined by four relative 

                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1



            # Output

                pooled_areas -- Tensor with the pooled region of interest, shape

                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)



        """

        def curried_pool_rois(x): 

          return ROIPoolingLayer._pool_rois(x[0], x[1], 

                                            self.pooled_height, 

                                            self.pooled_width)

        

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)



        return pooled_areas

    

    @staticmethod

    def _pool_rois(feature_map, rois, pooled_height, pooled_width):

        """ Applies ROI pooling for a single image and varios ROIs

        """

        def curried_pool_roi(roi): 

          return ROIPoolingLayer._pool_roi(feature_map, roi, 

                                           pooled_height, pooled_width)

        

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)

        return pooled_areas

    

    @staticmethod

    def _pool_roi(feature_map, roi, pooled_height, pooled_width):

        """ Applies ROI pooling to a single image and a single region of interest

        """



        # Compute the region of interest        

        feature_map_height = int(feature_map.shape[0])

        feature_map_width  = int(feature_map.shape[1])

        

        h_start = tf.cast(feature_map_height * roi[0], 'int32')

        w_start = tf.cast(feature_map_width  * roi[1], 'int32')

        h_end   = tf.cast(feature_map_height * roi[2], 'int32')

        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')

        

        region = feature_map[h_start:h_end, w_start:w_end, :]

        

        # Divide the region into non overlapping areas

        region_height = h_end - h_start

        region_width  = w_end - w_start

        h_step = tf.cast( region_height / pooled_height, 'int32')

        w_step = tf.cast( region_width  / pooled_width , 'int32')

        

        areas = [[(

                    i*h_step, 

                    j*w_step, 

                    (i+1)*h_step if i+1 < pooled_height else region_height, 

                    (j+1)*w_step if j+1 < pooled_width else region_width

                   ) 

                   for j in range(pooled_width)] 

                  for i in range(pooled_height)]

        

        # take the maximum of each area and stack the result

        def pool_area(x): 

          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

        

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

        return pooled_features