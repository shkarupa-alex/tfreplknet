import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfreplknet.block import Block


@keras_parameterized.run_all_keras_modes
class TestBlock(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            Block,
            kwargs={'kernel_size': 13, 'small_kernel': 5, 'ratio': 1., 'dropout': 0.},
            input_shape=[2, 64, 64, 3],
            input_dtype='float32',
            expected_output_shape=[None, 64, 64, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            Block,
            kwargs={'kernel_size': 31, 'small_kernel': None, 'ratio': 1.5, 'dropout': .2},
            input_shape=[2, 64, 64, 3],
            input_dtype='float32',
            expected_output_shape=[None, 64, 64, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
