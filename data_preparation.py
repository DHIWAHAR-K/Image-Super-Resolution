# data_preparation.py
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

def download_and_prepare_datasets():
    # Download DIV2K from TF Datasets using bicubic 4x degradation type
    div2k_data = tfds.image.Div2k(config="bicubic_x4")
    div2k_data.download_and_prepare()

    # Taking train data from div2k_data object
    train = div2k_data.as_dataset(split="train", as_supervised=True)
    train_cache = train.cache()

    # Validation data
    val = div2k_data.as_dataset(split="validation", as_supervised=True)
    val_cache = val.cache()

    return train_cache, val_cache