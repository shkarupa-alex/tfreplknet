from keras import activations, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfreplknet.large import LargeConv
from tfreplknet.drop import DropPath


@register_keras_serializable(package='TFRepLKNet')
class Block(layers.Layer):
    def __init__(self, kernel_size, small_kernel, ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.ratio = ratio
        self.dropout = dropout

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.bn = layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name='prelkb_bn')

        # noinspection PyAttributeOutsideInit
        self.pw1 = models.Sequential([
            layers.Conv2D(int(channels * self.ratio), 1, use_bias=False, name=f'{self.name}/pw1/conv'),
            layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'{self.name}/pw1/bn'),
            layers.ReLU()
        ], name='pw1')

        # noinspection PyAttributeOutsideInit
        self.large = LargeConv(self.kernel_size, self.small_kernel, name=f'{self.name}/large_kernel')

        # noinspection PyAttributeOutsideInit
        self.pw2 = models.Sequential([
            layers.Conv2D(channels, 1, use_bias=False, name=f'{self.name}/pw2/conv'),
            layers.BatchNormalization(momentum=0.1, epsilon=1.001e-5, name=f'{self.name}/pw2/bn')
        ], name='pw2')

        # noinspection PyAttributeOutsideInit
        self.drop = DropPath(self.dropout)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.bn(inputs)
        outputs = self.pw1(outputs)
        outputs = self.large(outputs)
        outputs = activations.relu(outputs)
        outputs = self.pw2(outputs)
        outputs = inputs + self.drop(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'small_kernel': self.small_kernel,
            'ratio': self.ratio,
            'dropout': self.dropout
        })

        return config
