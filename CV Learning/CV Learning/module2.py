import json
import pickle
from PIL import Image
import numpy as np

image = r"D:\Til data\train\train\17317.jpg"

x = Image.open(image)
print(np.array(x.resize((640,960))).shape)

with open(r"D:\Til data\images_val.p", "rb") as f:
    img_dict = pickle.load(f)

print(type(img_dict))
for key in img_dict:
    if(img_dict[key][0].shape != (640, 960, 3)):
        print(key)
        print(img_dict[key][0].shape)





