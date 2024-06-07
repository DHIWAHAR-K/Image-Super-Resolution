# train.py
import tensorflow as tf
from model import make_model
from tensorflow import keras
from data_augmentation import dataset_object
from data_preparation import download_and_prepare_datasets

def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio."""
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

train_cache, val_cache = download_and_prepare_datasets()

train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)

model = make_model(num_filters=64, num_of_residual_blocks=16)

optim_edsr = keras.optimizers.Adam(
    learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]
    )
)

model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])

model.fit(train_ds, epochs=100, steps_per_epoch=200, validation_data=val_ds)

model.save('model.h5')