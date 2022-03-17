from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFRepLKNet')
class LargeConv(layers.Layer):
    def __init__(self, kernel_size, small_kernel, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

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

        # noinspection PyAttributeOutsideInit
        self.small = models.Sequential([
            layers.DepthwiseConv2D(
                self.small_kernel, padding='same', use_bias=False, name=f'{self.name}/small_conv/conv'),
            layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'{self.name}/small_conv/bn')
        ], name='small_conv')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.big(inputs)
        outputs += self.small(inputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'small_kernel': self.small_kernel
        })

        return config
