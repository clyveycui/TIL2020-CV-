import os
import PIL
import pickle
import json
import numpy as np
from tqdm import tqdm
from math import log, exp
from random import shuffle  
from skimage.transform import resize
from PIL import ImageEnhance, ImageFont, ImageDraw, Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Layer

class class_net_sequence(Sequence):
    def __init__(self, x, y , batch_size, aug_fn, input_size=(224,224,3)):
        self.x = x
        self.y = y
        self.x_acc, self.y_acc = [], []
        self.batch_size = batch_size
        self.augment = aug_fn
        self.input_size = input_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def preprocess_npimg(self, x):
        return x * 1./255.

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        self.x_acc.clear()
        self.y_acc.clear()
        for x,y in zip( batch_x, batch_y ):
            x_img = Image.fromarray(x)
            x_aug = self.augment(x_img)
            self.x_acc.append(x_aug if x_aug.size == self.input_size else resize( x_aug, self.input_size[:2] ) )
            self.y_acc.append(y)

        return self.preprocess_npimg( np.array( self.x_acc ) ), np.array( self.y_acc )
  
#def aug_crop(img):
#    # Compute bounds such that no boxes are cut out
#    xmin, xmax, ymin, ymax = 0.25, 0.25, 0.75, 0.75
#    # Choose crop_xmin from [0, xmin]
#    crop_xmin = max( np.random.uniform() * xmin, 0 )
#    # Choose crop_xmax from [xmax, 1]
#    crop_xmax = min( xmax + (np.random.uniform() * (1-xmax)), 1 )
#    # Choose crop_ymin from [0, ymin]
#    crop_ymin = max( np.random.uniform() * ymin, 0 )
#    # Choose crop_ymax from [ymax, 1]
#    crop_ymax = min( ymax + (np.random.uniform() * (1-ymax)), 1 )
#    # Compute the "new" width and height of the cropped image
#    crop_w = crop_xmax - crop_xmin
#    crop_h = crop_ymax - crop_ymin
#    cropped_labels = []

#    W,H = img.size
#    # Compute the pixel coordinates and perform the crop
#    impix_xmin = int(W * crop_xmin)
#    impix_xmax = int(W * crop_xmax)
#    impix_ymin = int(H * crop_ymin)
#    impix_ymax = int(H * crop_ymax)
#    return img.crop( (impix_xmin, impix_ymin, impix_xmax, impix_ymax) )

def aug_translate(img):
    # Compute bounds such that no boxes are cut out
    xmin, xmax, ymin, ymax = 0.25, 0.75, 0.25, 0.75
    trans_range_x = [-xmin, 1 - xmax]
    tx = trans_range_x[0] + (np.random.uniform() * (trans_range_x[1] - trans_range_x[0]))
    trans_range_y = [-ymin, 1 - ymax]
    ty = trans_range_y[0] + (np.random.uniform() * (trans_range_y[1] - trans_range_y[0]))

    W,H = img.size
    tx_pix = int(W * tx)
    ty_pix = int(H * ty)
    return img.rotate(0, translate=(tx_pix, ty_pix))

def aug_colorbalance(img, color_factors=[0.2,2.0]):
    factor = color_factors[0] + np.random.uniform() * (color_factors[1] - color_factors[0])
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def aug_contrast(img, contrast_factors=[0.2,2.0]):
    factor = contrast_factors[0] + np.random.uniform() * (contrast_factors[1] - contrast_factors[0])
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def aug_brightness(img, brightness_factors=[0.2,2.0]):
    factor = brightness_factors[0] + np.random.uniform() * (brightness_factors[1] - brightness_factors[0])
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def aug_sharpness(img, sharpness_factors=[0.2,2.0]):
    factor = sharpness_factors[0] + np.random.uniform() * (sharpness_factors[1] - sharpness_factors[0])
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

# Performs no augmentations and returns the original image and bbox. Used for the validation images.
def aug_identity(pil_img):
    return np.array(pil_img)

# This is the default augmentation scheme that we will use for each training image.
def aug_default(img, p={'flip':0.5, 'crop':0.2, 'translate':0.2, 'color':0.2, 'contrast':0.2, 'brightness':0.2, 'sharpness':0.2}):
    if p['color'] > np.random.uniform():
        img = aug_colorbalance(img)
    if p['contrast'] > np.random.uniform():
        img = aug_contrast(img)
    if p['brightness'] > np.random.uniform():
        img = aug_brightness(img)
    if p['sharpness'] > np.random.uniform():
        img = aug_sharpness(img)
    if p['flip'] > np.random.uniform():
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    #if p['crop'] > np.random.uniform():
    #    img = aug_crop(img)
    if p['translate'] > np.random.uniform():
        img  = aug_translate(img)
    return np.array(img)


with open(r"D:\Til data\classification_network\cnn_x_448.p", "rb") as f:
    data_x = pickle.load(f)
with open(r"D:\Til data\classification_network\cnn_y_448.p", "rb") as f:
    cat_y = pickle.load(f)

data_y = np.zeros((len(cat_y),5))
for i,c in enumerate(cat_y):
    data_y[i,c-1] = 1
print(data_y.shape)
print(len(data_x))

model = tf.keras.applications.VGG19(include_top = False, input_shape = (448,448,3))

vgg_inputs = keras.Input(shape=(448,448,3))
x = model(vgg_inputs)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(2048)(x)
x = layers.Dense(2048, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(5,activation = "softmax")(x)
cnn = keras.Model(vgg_inputs, x,name="base_model")


cnn.compile( optimizer=keras.optimizers.Adam(learning_rate = 0.001),
                loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'] )
cnn.summary()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
train_sequence = class_net_sequence(train_x, train_y, batch_size =1, aug_fn = aug_default, input_size=(28,28,3))
test_sequence = class_net_sequence(test_x, test_y, batch_size =1, aug_fn = aug_identity, input_size=(28,28,3))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=r"D:\Til data\classification_network\best_cnn.h5",
                                                                save_weights_only=False,
                                                                monitor='val_loss',
                                                                mode='auto',
                                                                save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)

cnn.fit(x=train_sequence, 
        epochs=100, 
        validation_data=test_sequence,
        callbacks=[earlystopping, model_checkpoint_callback, reduce_lr],
        verbose = 1)

for layer in cnn.layers:
    layer.trainable = True

cnn.compile( optimizer=keras.optimizers.Adam(learning_rate = 0.0001),
                loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'] )


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=r"D:\Til data\classification_network\best_cnn_finetuned.h5",
                                                                save_weights_only=False,
                                                                monitor='val_loss',
                                                                mode='auto',
                                                                save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)

cnn.fit(x=train_sequence, 
        epochs=50, 
        validation_data=test_sequence,
        callbacks=[earlystopping, model_checkpoint_callback, reduce_lr],
        verbose = 1)

model.save(r"D:\Til data\classification_network\final_cnn.h5")