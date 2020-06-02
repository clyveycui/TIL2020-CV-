
def compute_reasonable_boundary(labels):
  bounds = [ (x-w/2, x+w/2, y-h/2, y+h/2) for _,x,y,w,h in labels]
  xmin = min([bb[0] for bb in bounds])
  xmax = min([bb[1] for bb in bounds])
  ymin = min([bb[2] for bb in bounds])
  ymax = min([bb[3] for bb in bounds])
  return xmin, xmax, ymin, ymax

def aug_horizontal_flip(img, labels):
  flipped_labels = []
  for c,x,y,w,h in labels:
    flipped_labels.append( (c,1-x,y,w,h) )
  return img.transpose(PIL.Image.FLIP_LEFT_RIGHT), np.array(flipped_labels)

def aug_crop(img, labels):
  # Compute bounds such that no boxes are cut out
  xmin, xmax, ymin, ymax = compute_reasonable_boundary(labels)
  # Choose crop_xmin from [0, xmin]
  crop_xmin = max( np.random.uniform() * xmin, 0 )
  # Choose crop_xmax from [xmax, 1]
  crop_xmax = min( xmax + (np.random.uniform() * (1-xmax)), 1 )
  # Choose crop_ymin from [0, ymin]
  crop_ymin = max( np.random.uniform() * ymin, 0 )
  # Choose crop_ymax from [ymax, 1]
  crop_ymax = min( ymax + (np.random.uniform() * (1-ymax)), 1 )
  # Compute the "new" width and height of the cropped image
  crop_w = crop_xmax - crop_xmin
  crop_h = crop_ymax - crop_ymin
  cropped_labels = []
  for c,x,y,w,h in labels:
    c_x = (x - crop_xmin) / crop_w
    c_y = (y - crop_ymin) / crop_h
    c_w = w / crop_w
    c_h = h / crop_h
    cropped_labels.append( (c,c_x,c_y,c_w,c_h) )

  W,H = img.size
  # Compute the pixel coordinates and perform the crop
  impix_xmin = int(W * crop_xmin)
  impix_xmax = int(W * crop_xmax)
  impix_ymin = int(H * crop_ymin)
  impix_ymax = int(H * crop_ymax)
  return img.crop( (impix_xmin, impix_ymin, impix_xmax, impix_ymax) ), np.array( cropped_labels )

def aug_translate(img, labels):
  # Compute bounds such that no boxes are cut out
  xmin, xmax, ymin, ymax = compute_reasonable_boundary(labels)
  trans_range_x = [-xmin, 1 - xmax]
  tx = trans_range_x[0] + (np.random.uniform() * (trans_range_x[1] - trans_range_x[0]))
  trans_range_y = [-ymin, 1 - ymax]
  ty = trans_range_y[0] + (np.random.uniform() * (trans_range_y[1] - trans_range_y[0]))

  trans_labels = []
  for c,x,y,w,h in labels:
    trans_labels.append( (c,x+tx,y+ty,w,h) )

  W,H = img.size
  tx_pix = int(W * tx)
  ty_pix = int(H * ty)
  return img.rotate(0, translate=(tx_pix, ty_pix)), np.array( trans_labels )

def aug_colorbalance(img, labels, color_factors=[0.2,2.0]):
  factor = color_factors[0] + np.random.uniform() * (color_factors[1] - color_factors[0])
  enhancer = ImageEnhance.Color(img)
  return enhancer.enhance(factor), labels

def aug_contrast(img, labels, contrast_factors=[0.2,2.0]):
  factor = contrast_factors[0] + np.random.uniform() * (contrast_factors[1] - contrast_factors[0])
  enhancer = ImageEnhance.Contrast(img)
  return enhancer.enhance(factor), labels

def aug_brightness(img, labels, brightness_factors=[0.2,2.0]):
  factor = brightness_factors[0] + np.random.uniform() * (brightness_factors[1] - brightness_factors[0])
  enhancer = ImageEnhance.Brightness(img)
  return enhancer.enhance(factor), labels

def aug_sharpness(img, labels, sharpness_factors=[0.2,2.0]):
  factor = sharpness_factors[0] + np.random.uniform() * (sharpness_factors[1] - sharpness_factors[0])
  enhancer = ImageEnhance.Sharpness(img)
  return enhancer.enhance(factor), labels

# Performs no augmentations and returns the original image and bbox. Used for the validation images.
def aug_identity(pil_img, label_arr):
  return np.array(pil_img), label_arr

# This is the default augmentation scheme that we will use for each training image.
def aug_default(img, labels, p={'flip':0.5, 'crop':0.2, 'translate':0.2, 'color':0.2, 'contrast':0.2, 'brightness':0.2, 'sharpness':0.2}):
  if p['color'] > np.random.uniform():
    img, labels = aug_colorbalance(img, labels)
  if p['contrast'] > np.random.uniform():
    img, labels = aug_contrast(img, labels)
  if p['brightness'] > np.random.uniform():
    img, labels = aug_brightness(img, labels)
  if p['sharpness'] > np.random.uniform():
    img, labels = aug_sharpness(img, labels)
  if p['flip'] > np.random.uniform():
    img, labels = aug_horizontal_flip(img, labels)
  if p['crop'] > np.random.uniform():
    img, labels = aug_crop(img, labels)
  if p['translate'] > np.random.uniform():
    img, labels = aug_translate(img, labels)
  return np.array(img), labels