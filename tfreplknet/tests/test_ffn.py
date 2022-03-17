import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfreplknet.ffn import FFN


@keras_parameterized.run_all_keras_modes
class TestFFN(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            FFN,
            kwargs={'ratio': .5, 'dropout': 0.},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            FFN,
            kwargs={'ratio': 1.5, 'dropout': .2},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
