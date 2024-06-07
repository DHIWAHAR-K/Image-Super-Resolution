#utils.py
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_images(high, low, predicted):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('High Image', color='green', fontsize=20)
    plt.imshow(high)
    plt.subplot(1, 3, 2)
    plt.title('Low Image', color='black', fontsize=20)
    plt.imshow(low)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Image', color='Red', fontsize=20)
    plt.imshow(predicted)
    plt.show()

def downsample_image(image, scale):
    x = tf.image.resize(image / 255, (image.shape[0] // scale, image.shape[1] // scale))
    x = tf.image.resize(x, (image.shape[0], image.shape[1]), method=tf.image.ResizeMethod.BICUBIC)
    return x