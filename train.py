#train.py
from data_loader import get_datasets
from model import build_unet, compile_model
from tensorflow.keras.utils import plot_model

def train_model():
    train_high_image, train_low_image, val_high_image, val_low_image, _, _ = get_datasets()
    model = build_unet((256, 256, 3))
    model = compile_model(model)
    plot_model(model, to_file='super_res(U-NET).png', show_shapes=True)
    model.fit(train_low_image, train_high_image, epochs=10, batch_size=1,
              validation_data=(val_low_image, val_high_image))
    model.save('superresolution_unet.h5')

if __name__ == "__main__":
    train_model()
