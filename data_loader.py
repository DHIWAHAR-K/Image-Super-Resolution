#data_loader.py
import os
import re
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

SIZE = 256

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def load_images(path, limit=None):
    images = []
    files = os.listdir(path)
    files = sorted_alphanumeric(files)
    for i in tqdm(files):
        if limit and i == limit:
            break
        img = cv2.imread(os.path.join(path, i), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        images.append(img_to_array(img))
    return images

def get_datasets():
    high_path = '../input/image-super-resolution/dataset/Raw Data/high_res'
    low_path = '../input/image-super-resolution/dataset/Raw Data/low_res'
    
    high_img = load_images(high_path, '855.png')
    low_img = load_images(low_path, '855.png')
    
    train_high_image = np.reshape(high_img[:700], (700, SIZE, SIZE, 3))
    train_low_image = np.reshape(low_img[:700], (700, SIZE, SIZE, 3))
    validation_high_image = np.reshape(high_img[700:810], (110, SIZE, SIZE, 3))
    validation_low_image = np.reshape(low_img[700:810], (110, SIZE, SIZE, 3))
    test_high_image = np.reshape(high_img[810:], (45, SIZE, SIZE, 3))
    test_low_image = np.reshape(low_img[810:], (45, SIZE, SIZE, 3))
    
    return train_high_image, train_low_image, validation_high_image, validation_low_image, test_high_image, test_low_image
