from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf
import img_aug
import numpy as np

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

class TILSequence(Sequence):
  def __init__(self, img_folder, json_annotation_file, batch_size, augment_fn, input_size, label_encoder, preprocess_fn, testmode=False):

    self._prepare_data(img_folder, json_annotation_file)
    self.batch_size = batch_size
    self.augment_fn = augment_fn
    self.input_wh = (*input_size[:2][::-1],input_size[2])
    self.label_encoder = label_encoder
    self.preprocess_fn = preprocess_fn
    self.testmode = testmode
    
  def _prepare_data(self, img_folder, json_annotation_file):
    imgs_dict = {im.split('.')[0]:im for im in os.listdir(img_folder) if im.endswith('.jpg')}
    data_dict = {}
    with open(json_annotation_file, 'r') as f:
      annotations_dict = json.load(f)
    annotations_list = annotations_dict['annotations']
    for annotation in annotations_list:
      img_id = str(annotation['image_id'])
      c = annotation['category_id'] - 1 # TODO: make sure that category ids start from 1, not 0
      boxleft,boxtop,boxwidth,boxheight = annotation['bbox']
      if img_id in imgs_dict:
        img_fp = os.path.join(img_folder, imgs_dict[img_id])
        imwidth,imheight = PIL.Image.open(img_fp).size
        if img_id not in data_dict:
          data_dict[img_id] = []
        box_cenx = boxleft + boxwidth/2.
        box_ceny = boxtop + boxheight/2.
        x,y,w,h = box_cenx/imwidth, box_ceny/imheight, boxwidth/imwidth, boxheight/imheight

        data_dict[img_id].append( [c,x,y,w,h] )
    self.x, self.y, self.ids = [], [], []
    for img_id, labels in data_dict.items():
      self.x.append( os.path.join(img_folder, imgs_dict[img_id]) )
      self.y.append( np.array(labels) )
      self.ids.append( img_id )

  def __len__(self):
    return int(np.ceil(len(self.x) / float(self.batch_size)))
  
  def __getitem__(self, idx):
    return self.get_batch_test(idx) if self.testmode else self.get_batch(idx)

  def get_batch_test(self, idx):
    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]

    x_acc, y_acc = [], {}
    original_img_dims = []
    with Pool(self.batch_size) as p:
      # Read in the PIL objects from filepaths
      batch_x = p.map(load_img, batch_x)
    
    for x,y in zip( batch_x, batch_y ):
      W,H = x.size
      original_img_dims.append( (W,H) )
      x_aug, y_aug = self.augment_fn( x, y )
      if x_aug.size != self.input_wh[:2]:
        x_aug.resize( self.input_wh )
      x_acc.append( np.array(x_aug) )
      y_dict = self.label_encoder( y_aug )
      for dimkey, label in y_dict.items():
        if dimkey not in y_acc:
          y_acc[dimkey] = []
        y_acc[dimkey].append( label )

    return batch_ids, original_img_dims, self.preprocess_fn( np.array( x_acc ) ), { dimkey: np.array( gt_tensor ) for dimkey, gt_tensor in y_acc.items() }

  def get_batch(self, idx):
    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

    x_acc, y_acc = [], {}
    with Pool(self.batch_size) as p:
      # Read in the PIL objects from filepaths
      batch_x = p.map(load_img, batch_x)
    
    for x,y in zip( batch_x, batch_y ):
      x_aug, y_aug = self.augment_fn( x, y )
      if x_aug.size != self.input_wh[:2]:
        x_aug.resize( self.input_wh )
      x_acc.append( np.array(x_aug) )
      y_dict = self.label_encoder( y_aug )
      for dimkey, label in y_dict.items():
        if dimkey not in y_acc:
          y_acc[dimkey] = []
        y_acc[dimkey].append( label )

    return self.preprocess_fn( np.array( x_acc ) ), { dimkey: np.array( gt_tensor ) for dimkey, gt_tensor in y_acc.items() }