from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfreplknet.ffn import FFN
from tfreplknet.block import Block


@register_keras_serializable(package='TFRepLKNet')
class Stage(layers.Layer):
    def __init__(self, kernel_size, small_kernel, dw_ratio, ffn_ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if not isinstance(dropout, (list, tuple)):
            raise ValueError(f'The "dropout" argument must be a list of floats. Got: {dropout}')

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.dw_ratio = dw_ratio
        self.ffn_ratio = ffn_ratio
        self.dropout = dropout

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.blocks = models.Sequential()
        for i, dropout in enumerate(self.dropout):
            self.blocks.add(Block(
                self.kernel_size, self.small_kernel, self.dw_ratio, dropout, name=f'{self.name}/blocks/{i * 2}'))
            self.blocks.add(FFN(self.ffn_ratio, dropout, name=f'{self.name}/blocks/{i * 2 + 1}'))

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.blocks(inputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'small_kernel': self.small_kernel,
            'dw_ratio': self.dw_ratio,
            'ffn_ratio': self.ffn_ratio,
            'dropout': self.dropout
        })

        return config
