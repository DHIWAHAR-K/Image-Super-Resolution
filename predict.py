# predict.py
import tensorflow as tf
from model import make_model
import matplotlib.pyplot as plt
from data_augmentation import dataset_object
from data_preparation import download_and_prepare_datasets

train_cache, val_cache = download_and_prepare_datasets()
val_ds = dataset_object(val_cache, training=False)

model = make_model(num_filters=64, num_of_residual_blocks=16)
model = tf.keras.models.load_model('path_t')

def plot_results(lowres, preds):
    """Displays low resolution image and super resolution image."""
    plt.figure(figsize=(24, 14))
    plt.subplot(132), plt.imshow(lowres), plt.title("Low resolution")
    plt.subplot(133), plt.imshow(preds), plt.title("Prediction")
    plt.show()

for lowres, highres in val_ds.take(10):
    lowres = tf.image.random_crop(lowres, (150, 150, 3))
    preds = model.predict_step(lowres)
    plot_results(lowres, preds)