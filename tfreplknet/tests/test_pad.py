import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfreplknet.pad import SamePad


@keras_parameterized.run_all_keras_modes
class TestSamePad(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SamePad,
            kwargs={'kernel_size': 7, 'dilation_rate': 1},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 22, 22, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            SamePad,
            kwargs={'kernel_size': 3, 'dilation_rate': 3},
            input_shape=[2, 19, 19, 3],
            input_dtype='float32',
            expected_output_shape=[None, 25, 25, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
