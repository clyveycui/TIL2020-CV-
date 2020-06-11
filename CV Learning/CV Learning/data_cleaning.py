from PIL import Image
import os
import pickle
import numpy as np
import json

jsonf = open(r'D:\til2020\train.json')
data = json.load(jsonf)
jsonf.close()

img_directory = r'D:\til2020\train\train'
train_img_path = []
for entry in os.scandir(img_directory):
    if (entry.path.endswith(".jpg")) and entry.is_file():
        train_img_path.append(entry.path)
images = []
for i,path in enumerate(train_img_path):
    img = Image.open(path)
    print(path)
    img_resized = np.array(img.resize((960,640)))
    img_id = data["images"][i]['id']
    images.append([img_resized, img_id])
    img.close()

filename = r"D:\til2020\train.p\images.p"
outfile = open(filename, "wb")
pickle.dump(images, outfile)
outfile.close()



