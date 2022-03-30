import tensorflow as tf
from keras import backend, layers, models
from keras.mixed_precision.autocast_variable import AutoCastVariable
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFRepLKNet')
class LargeConv(layers.Layer):
    def __init__(self, kernel_size, small_kernel, fused=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.fused = fused

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.big = models.Sequential([
            layers.DepthwiseConv2D(
                self.kernel_size, padding='same', use_bias=False, name=f'{self.name}/lkb_origin/conv'),
            layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'{self.name}/lkb_origin/bn')
        ], name='lkb_origin')
        self.big.build(input_shape)

        if self.small_kernel is not None:
            # noinspection PyAttributeOutsideInit
            self.small = models.Sequential([
                layers.DepthwiseConv2D(
                    self.small_kernel, padding='same', use_bias=False, name=f'{self.name}/small_conv/conv'),
                layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'{self.name}/small_conv/bn')
            ], name='small_conv')
            self.small.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=None, *args, **kwargs):
        if self.small_kernel is None or not self.fused:
            return self._branched(inputs)

        if not self.trainable:
            training = False
        elif training is None:
            training = backend.learning_phase()

        outputs = smart_cond(training, lambda: self._branched(inputs), lambda: self._merged(inputs))

        return outputs

    def _branched(self, inputs):
        outputs = self.big(inputs)

        if self.small_kernel is not None:
            outputs += self.small(inputs)

        return outputs

    def _merged(self, inputs):
        def _raw_var(var):
            return var if not isinstance(var, AutoCastVariable) else var._variable

        def _fuse_conv_bn(conv, bn):
            fused_scale = bn.gamma / tf.sqrt(bn.moving_variance + bn.epsilon)
            fused_kernel = _raw_var(conv.depthwise_kernel) * fused_scale[None, None, ..., None]

            fused_bias = bn.beta - bn.moving_mean * fused_scale

            return fused_kernel, fused_bias

        big_kernel, big_bias = _fuse_conv_bn(*self.big.layers)
        small_kernel, small_bias = _fuse_conv_bn(*self.small.layers)

        small_pad = [[(self.kernel_size - self.small_kernel) // 2] * 2] * 2 + [[0, 0]] * 2
        kernel = big_kernel + tf.pad(small_kernel, small_pad)
        kernel = tf.cast(kernel, self.compute_dtype)

        bias = big_bias + small_bias
        bias = tf.cast(bias, self.compute_dtype)

        outputs = backend.depthwise_conv2d(inputs, kernel, padding='same', data_format=backend.image_data_format())
        outputs = backend.bias_add(outputs, bias, data_format=backend.image_data_format())

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'small_kernel': self.small_kernel,
            'fused': self.fused
        })

        return config
