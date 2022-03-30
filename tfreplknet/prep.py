import tensorflow as tf
from keras.applications import imagenet_utils


def preprocess_input_bl(inputs):
    inputs = tf.cast(inputs, 'float32')
    outputs = imagenet_utils.preprocess_input(inputs, data_format='channels_last', mode='torch')

    return outputs


def preprocess_input_xl(inputs):
    inputs = tf.cast(inputs, 'float32')
    outputs = imagenet_utils.preprocess_input(inputs, data_format='channels_last', mode='tf')

    return outputs
