from keras import layers
from keras.utils.conv_utils import normalize_tuple
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFRepLKNet')
class SamePad(layers.Layer):
    def __init__(self, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')
        self.dilation_rate = normalize_tuple(dilation_rate, 2, 'dilation_rate')

    @shape_type_conversion
    def build(self, input_shape):
        total_pad = (self.kernel_size[0] - 1) * self.dilation_rate[0], \
                    (self.kernel_size[1] - 1) * self.dilation_rate[1]

        top_pad = total_pad[0] // 2
        bottom_pad = total_pad[0] - top_pad
        left_pad = total_pad[1] // 2
        right_pad = total_pad[1] - top_pad

        # noinspection PyAttributeOutsideInit
        self.pad = layers.ZeroPadding2D(((top_pad, bottom_pad), (left_pad, right_pad)))

        super().build(input_shape)

    def call(self, inputs):
        return self.pad(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.pad.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })

        return config
