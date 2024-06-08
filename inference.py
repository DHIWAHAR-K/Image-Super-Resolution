#inference.py

import tensorflow as tf
from model import make_model
from download_data import val
import matplotlib.pyplot as plt

# Create model and load weights
model = make_model(num_filters=64, num_of_residual_blocks=16)
model.load_weights("model/model.h5")

def plot_results(lowres, preds):
    """
    Displays low resolution image and super resolution image
    """
    plt.figure(figsize=(24, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(lowres)
    plt.title("Low resolution")
    plt.subplot(1, 2, 2)
    plt.imshow(preds)
    plt.title("Prediction")
    plt.show()

# Ensure you get data from the correct source
for lowres, highres in val.take(10):
    lowres_cropped = tf.image.random_crop(lowres, (150, 150, 3))
    preds = model.predict_step(lowres_cropped)
    plot_results(lowres_cropped, preds)