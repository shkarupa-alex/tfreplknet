import numpy as np
import tensorflow as tf
from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.utils import data_utils, layer_utils
from tfreplknet.pad import SamePad
from tfreplknet.stage import Stage

BASE_URL = 'https://github.com/shkarupa-alex/tfreplknet/releases/download/1.0.0/{}.h5'
WEIGHT_HASHES = {
    'rep_l_k_net_31_b_224_k1': 'b25e86d899e1fdf6910bc257edd681d2fbad2e963220b4886c6b0690328927d5',
    'rep_l_k_net_31_b_224_k21': '65dc869425664d76909c12257d907015302681c68328349030c4e2e13eff1b63',
    'rep_l_k_net_31_b_384_k1': 'f0ee368fdfbf3302ea07323fa5aa4c24f6612047327916b0b9ace40ab9d41bec',
    # 'rep_l_k_net_31_b_384_k21': '',
    'rep_l_k_net_31_l_384_k1': '4c5e87d8734b7a5e0e15955966c37e68de77a9fe4f573f7f78b992a0194367df',
    'rep_l_k_net_31_l_384_k21': '5e616b0a5dbd67119ed1c5b2ee037bfbb5c360507689355fbf6e0240a4b47c62',
}


def RepLKNet(filters, kernel_sizes=(31, 29, 27, 13), small_kernel=5, depths=(2, 2, 18, 2), ffn_ratio=4, path_drop=0.3,
             model_name='rep_l_k_net', include_top=True, weights=None, input_tensor=None, input_shape=None,
             pooling=None, classes=1000, classifier_activation='softmax'):
    """Instantiates the Re-parameterized Large Kernel Network architecture.

    Args:
      filters: features size for different stages.
      kernel_sizes: large kernel size for different stages.
      small_kernel: small kernel size for all stages.
      depths: depth of each stage.
      ffn_ratio: ratio of feed-forward hidden units to embedding units.
      path_drop: stochastic depth rate.
      model_name: model name.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet or ImageNet 21k), or the
        path to the weights file to be loaded.
      input_tensor: tensor (i.e. output of `layers.Input()`) to use as image input for the model.
      input_shape: shape tuple without batch dimension. Used to create input layer if `input_tensor` not provided.
      pooling: optional pooling mode for feature extraction when `include_top` is `False`.
        - `None` means that the output of the model will be the 3D tensor output of the last layer.
        - `avg` means that global average pooling will be applied to the output of the last layer, and thus the output
          of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True.
      classifier_activation: the activation function to use on the "top" layer. Ignored unless `include_top=True`.
        When loading pretrained weights, `classifier_activation` can only be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 21841}:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000 or 21841 depending on model type')

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor is not None:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=384,
        min_size=32,
        data_format='channel_last',
        require_flatten=False,
        weights=weights)

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype='float32')
    else:
        image = layers.Input(shape=input_shape)

    x = image

    x = SamePad(3)(x)
    x = layers.Conv2D(filters[0], 3, strides=2, use_bias=False, name='stem/0/conv')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name='stem/0/bn')(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False, name='stem/1/conv')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name='stem/1/bn')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters[0], 1, use_bias=False, name='stem/2/conv')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name='stem/2/bn')(x)
    x = layers.ReLU()(x)

    x = SamePad(3)(x)
    x = layers.DepthwiseConv2D(3, strides=2, use_bias=False, name='stem/3/conv')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name='stem/3/bn')(x)
    x = layers.ReLU()(x)

    path_drops = np.linspace(0., path_drop, sum(depths))

    for i in range(len(depths)):
        path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])].tolist()
        not_last = i != len(depths) - 1

        x = Stage(
            kernel_size=kernel_sizes[i], small_kernel=small_kernel, ratio=ffn_ratio, dropout=path_drop,
            name=f'stages/{i}')(x)
        if not_last:
            x = layers.Conv2D(filters[i + 1], 1, use_bias=False, name=f'transitions/{i}/0/conv')(x)
            x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'transitions/{i}/0/bn')(x)
            x = layers.ReLU()(x)

            x = SamePad(3)(x)
            x = layers.DepthwiseConv2D(3, strides=2, use_bias=False, name=f'transitions/{i}/1/conv')(x)
            x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'transitions/{i}/1/bn')(x)
            x = layers.ReLU()(x)

    x = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name='norm')(x)

    if include_top or pooling in {None, 'avg'}:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    else:
        raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {pooling}')

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, name='head')(x)
    x = layers.Activation(classifier_activation, dtype='float32', name='pred')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if 'imagenet' == weights and model_name in WEIGHT_HASHES:
        weights_url = BASE_URL.format(model_name)
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfreplknet')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    last_layer = 'norm'
    if pooling == 'avg':
        last_layer = 'avg_pool'
    elif pooling == 'max':
        last_layer = 'max_pool'

    outputs = model.get_layer(name=last_layer).output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


# Architectures

def RepLKNet31B(model_name, classes, filters=(128, 256, 512, 1024), weights='imagenet', **kwargs):
    return RepLKNet(model_name=model_name, filters=filters, weights=weights, classes=classes, **kwargs)


def RepLKNet31L(model_name, classes, filters=(192, 384, 768, 1536), weights='imagenet', **kwargs):
    return RepLKNet(model_name=model_name, filters=filters, weights=weights, classes=classes, **kwargs)


# Weights

def RepLKNet31B224K1(model_name='rep_l_k_net_31_b_224_k1', classes=1000, **kwargs):
    return RepLKNet31B(model_name=model_name, classes=classes, **kwargs)


def RepLKNet31B224K21(model_name='rep_l_k_net_31_b_224_k21', classes=21841, **kwargs):
    return RepLKNet31B(model_name=model_name, classes=classes, **kwargs)


def RepLKNet31B384K1(model_name='rep_l_k_net_31_b_384_k1', classes=1000, **kwargs):
    return RepLKNet31B(model_name=model_name, classes=classes, **kwargs)


# def RepLKNet31B384K21(model_name='rep_l_k_net_31_b_384_k21', classes=21841, **kwargs):
#     return RepLKNet31B(model_name=model_name, classes=classes, **kwargs)

def RepLKNet31L384K1(model_name='rep_l_k_net_31_l_384_k1', classes=1000, **kwargs):
    return RepLKNet31L(model_name=model_name, classes=classes, **kwargs)


def RepLKNet31L384K21(model_name='rep_l_k_net_31_l_384_k21', classes=21841, **kwargs):
    return RepLKNet31L(model_name=model_name, classes=classes, **kwargs)
