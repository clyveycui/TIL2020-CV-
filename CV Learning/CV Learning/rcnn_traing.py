import tensorflow as tf
import data_gen
import numpy as np
from roi_pooling import ROIPoolingLayer
from data_gen import generate_anchor_boxes
from tensorflow import keras
from tensorflow.keras import layers
import pickle

def masked_regr_loss(ytrue, ypred):
    mask = tf.where(ytrue==0, 0., 1.)
    ypred = tf.multiply(mask,ypred)
    return keras.losses.Huber()(ytrue,ypred)



#rpn_model_path = r"D:\Til data\rpn_model_finetuned.h5"
#rpn = tf.keras.models.load_model(rpn_model_path, custom_objects={'custom_loss':rpn_loss})
#rpn.trainable = False

input_shape = (480,480, 3)
anchor_boxes = generate_anchor_boxes(480,480,16,scale=[64,128,256])


model = tf.keras.applications.vgg16.VGG16(input_shape = (480,480,3),weights="imagenet", include_top=False)
model.layers.pop()
model_2 = tf.keras.models.Sequential(name="VGG16")
for layer in model.layers[:-1]:
    model_2.add(layer)
for layer in model_2.layers:
    layer.trainable = False

inputs = keras.Input(shape = input_shape, name="image")
fm = model_2(inputs)
#rpn_o = rpn(inputs)
#rp = RegionProposalLayer(anchor_boxes)(rpn_o)

rp = keras.Input(shape =(12,4), name="region_proposals")
#Need to convert rp into a tensor of shape (batch_size, n_rois, 4) in the form of (xmin, ymin, xmax, ymax)
x = ROIPoolingLayer(pooled_height =3,pooled_width=3)([fm, rp])
x = tf.keras.layers.AveragePooling3D(pool_size = (1,3,3))(x)
x = tf.keras.layers.Dense(2048)(x)
x = tf.keras.layers.Dense(2048)(x)
object_classification = tf.keras.layers.Dense(6)(x)
object_classification = layers.Softmax(axis=-1, name="c")(object_classification)
bbox_regr = tf.keras.layers.Dense(24, name="r")(x)
rcnn = tf.keras.Model(inputs = [inputs,rp], outputs=[object_classification, bbox_regr], name="rcnn")
rcnn.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss={"c":tf.keras.losses.CategoricalCrossentropy(), "r":masked_regr_loss},
            metrics=["Accuracy"])

rcnn._layers = [layer for layer in rcnn._layers if not isinstance(layer, dict)]

model_img_f = "D:\\Programming\\Python\\DSTA CV\\rcnn.png"
keras.utils.plot_model(rcnn, to_file=model_img_f, show_shapes=True)

class_y = r"D:\Til data\rcnn\rcnn_class_y.p"
regr_y = r"D:\Til data\rcnn\rcnn_regr_y.p"
img_x = r"D:\Til data\rcnn\rcnn_img_x.p"
rp_x = r"D:\Til data\rcnn\rcnn_rp_x.p"

with open(class_y, "rb") as f:
    rcnn_class_y = pickle.load(f)
with open(regr_y, "rb") as f:
    rcnn_regr_y = pickle.load(f)
with open(img_x, "rb") as f:
    rcnn_img_x = pickle.load(f)
with open(rp_x, "rb") as f:
    rcnn_rp_x = pickle.load(f)

count = [0,0,0,0,0,0]
for i in range(rcnn_class_y.shape[0]):
    for j in range(rcnn_class_y.shape[1]):
        count = np.add(count, rcnn_class_y[i,j,0,0])

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
mcp_save = keras.callbacks.ModelCheckpoint(r"D:\Til data\rcnn\best_rcnn.h5", save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

rcnn.fit(x={"image":rcnn_img_x, "region_proposals":rcnn_rp_x},
        y={"c":rcnn_class_y, "r":rcnn_regr_y}, 
        batch_size=2,
        epochs=100, 
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        validation_split = 0.2,
        verbose = 2)

del rcnn
rcnn = tf.keras.models.load_model(r"D:\Til data\rcnn\best_rcnn.h5", custom_objects = {"masked_regr_loss":masked_regr_loss,"ROIPoolingLayer" :ROIPoolingLayer})
for layer in rcnn.layers:
    layer.trainable = True
rcnn.summary()


earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
mcp_save = keras.callbacks.ModelCheckpoint(r"D:\Til data\rcnn\best_rcnn_finetuned.h5", save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

rcnn.fit(x={"image":rcnn_img_x, "region_proposals":rcnn_rp_x},
        y={"c":rcnn_class_y, "r":rcnn_regr_y}, 
        batch_size=2,
        epochs=50, 
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        validation_split = 0.2,
        verbose = 1)

rcnn_model_path = r"D:\Til data\rcnn\rcnn_final.h5"
rcnn.save(rcnn_model_path)