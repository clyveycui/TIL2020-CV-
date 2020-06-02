from tensorflow.python.keras.utils.data_utils import Sequence
import img_aug

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

class ImageVocSequence(Sequence):
    def __init__(self, dataset, batch_size, augmentations, dims, input_size=(224,224,3), iou_fn=iou):
        self.x, self.y = zip(*dataset)
        self.x_acc, self.y_acc = [], []
        self.batch_size = batch_size
        self.augment = augmentations
        self.dims = dims
        self.input_size = input_size
        self.iou = iou_fn

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
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

    def non_max_suppression(boxes, ):
        return None

    def preprocess_anchor_boxes(self,anchor_boxes):
        arr_shape = anchor_boxes.shape
        
    


    def create_ytrue_train(self, labels, anchor_boxes, iou):
        gtclass, gtx, gty, gtw, gth = label
        gtclass = int(gtclass)
        gt_bbox = [gtx, gty, gtw, gth]
        anchor

    #def convert_labels_cxywh_to_arrays(self, labels, iou_threshold=0.5, exceed_thresh_positive=True):
    #    num_entries = 7 # objectness, p_cat, p_dog, dx, dy, dw, dh
    #    kx,ky = self.dims
    #    labels_arr = np.zeros( (kx, ky, num_entries) ) # For this basic model, this is of shape (3,3,7)

    #    for label in labels:
    #    # Retrieve the ground-truth class label and bbox
    #        gtclass, gtx, gty, gtw, gth = label
    #        gtclass = int(gtclass)
    #        gt_bbox = [gtx, gty, gtw, gth]
      
    #    iou_scores = []

    #    '''
    #    There are kx x ky cells. In the basic model, this is 3x3.
    #    Each cell is of width=gapx and height=gapy
    #    For the (i,j)-th tile, center-x = (0.5+i)*gapx | center-y = (0.5+j)*gapy
    #    '''
    #    gapx = 1.0 / kx
    #    gapy = 1.0 / ky
    #    # In this loop, we run through all cells of the 3x3 grid, compute the intersection-over-union w the ground-truth and also the targets to predict.
    #    for i in range(kx):
    #        for j in range(ky):
    #            x = (0.5+i)*gapx
    #            y = (0.5+j)*gapy

    #            # These are fixed to the width and height of the square-cell at the moment. However, if we want more anchor boxes of varying aspect ratios, this is the place to change it.
    #            w = gapx
    #            h = gapy
    #            candidate_bbox = [x,y,w,h]

    #            # Based on the SSD training regime. These are the targets we wish for the CNN to predict at the end.
    #            # Read the SSD paper: https://arxiv.org/pdf/1512.02325.pdf, for more details.
    #            dx = (gtx - x) / w 
    #            dy = (gty - y) / h
    #            dw = log( gtw / w )
    #            dh = log( gth / h )

    #            IoU = self.iou( candidate_bbox, gt_bbox )
    #            iou_scores.append( (IoU, i, j, dx, dy, dw, dh) )
    #    # Sort by IoU: only the highest IoU scores get included into the resulting label array. Cutoff at threshold.
    #    iou_scores.sort( key=lambda x: x[0], reverse=True )
    #    # Count the top 25% of iou scores
    #    top_count = max( round(len(iou_scores) * 0.25), 1)
    #    # Remove all the grid cells that do not overlap with ground truth at all
    #    iou_scores = [iou_score for iou_score in iou_scores if iou_score[0] > 0]
    #    iou_scores = iou_scores[:top_count] + [iou_score for iou_score in iou_scores[top_count:] if iou_score[0] >= iou_threshold]
    #    # Always take the top IoU entry
    #    for iou_score in iou_scores:
    #        # The top IoU score is always included
    #        IoU, i, j, dx, dy, dw, dh = iou_score
    #        payload = [IoU, 0, 0, dx,dy,dw,dh]
    #        payload[gtclass + 1] = 1
    #        labels_arr[i,j,:] = payload
    #    return labels_arr
    
    def preprocess_npimg(self, x):
          return x * 1./255.

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        self.x_acc.clear()
        self.y_acc.clear()
        for x,y in zip( batch_x, batch_y ):
            x_aug, y_aug = self.augment( x, y )
            self.x_acc.append( x_aug if x_aug.shape == self.input_size else resize( x_aug, self.input_size[:2] ) )
            y_arr = self.convert_labels_cxywh_to_arrays( y_aug )
            self.y_acc.append( y_arr )

        return self.preprocess_npimg( np.array( self.x_acc ) ), np.array( self.y_acc )