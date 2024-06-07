#evaluate.py
import numpy as np
import tensorflow as tf
from utils import plot_images
import matplotlib.pyplot as plt
from model import pixel_mse_loss
from data_loader import get_datasets

def PSNR(y_true, y_pred):
    mse = tf.reduce_mean((y_true - y_pred) ** 2)
    return 20 * log10(1 / (mse ** 0.5))

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def evaluate_model():
    _, _, _, _, test_high_image, test_low_image = get_datasets()
    model = tf.keras.models.load_model('superresolution_unet.h5', custom_objects={'pixel_mse_loss': pixel_mse_loss})
    
    for i in range(16, 25):
        predicted = np.clip(model.predict(test_low_image[i].reshape(1, 256, 256, 3)), 0.0, 1.0).reshape(256, 256, 3)
        plot_images(test_high_image[i], test_low_image[i], predicted)
        print('PSNR', PSNR(test_high_image[i], predicted), 'dB')

if __name__ == "__main__":
    evaluate_model()