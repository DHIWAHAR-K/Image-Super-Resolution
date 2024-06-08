#metrics.py

import tensorflow as tf

def PSNR(super_resolution, high_resolution):
    
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value