import tensorflow as tf
import tfreplknet
from absl.testing import parameterized
from keras import preprocessing
from keras.applications import imagenet_utils
from keras.utils import data_utils

MODEL_LIST = [
    (tfreplknet.RepLKNet31B224K1, 224, 1024),
    (tfreplknet.RepLKNet31B224K21, 224, 1024),
    (tfreplknet.RepLKNet31B384K1, 384, 1024),
    # (tfreplknet.RepLKNet31B384K21, 384, 1024),
    (tfreplknet.RepLKNet31L384K1, 384, 1536),
    (tfreplknet.RepLKNet31L384K21, 384, 1536),
]


class ApplicationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, *_):
        # Can be instantiated with default arguments
        model = app(weights=None)

        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)

        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_notop(self, app, _, last_dim):
        output_shape = app(weights=None, include_top=False).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, _, last_dim):
        output_shape = app(weights=None, include_top=False, pooling='avg').output_shape
        self.assertLen(output_shape, 2)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_input_1_channel(self, app, size, last_dim):
        input_shape = (size, size, 1)
        output_shape = app(weights=None, include_top=False, input_shape=input_shape).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_input_4_channels(self, app, size, last_dim):
        input_shape = (size, size, 4)
        output_shape = app(weights=None, include_top=False, input_shape=input_shape).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_weights_notop(self, app, _, last_dim):
        model = app(weights='imagenet', include_top=False)
        self.assertEqual(model.output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_predict(self, app, size, _):
        model = app(weights='imagenet')
        self.assertIn(model.output_shape[-1], {1000, 21841})

        test_image = data_utils.get_file(
            'elephant.jpg', 'https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg')
        image = preprocessing.image.load_img(test_image, target_size=(size, size), interpolation='bicubic')
        image = preprocessing.image.img_to_array(image)[None, ...]

        image_ = tfreplknet.preprocess_input(image)
        preds = model.predict(image_)

        if 1000 == preds.shape[-1]:
            names = [p[1] for p in imagenet_utils.decode_predictions(preds, top=1)[0]]

            # Test correct label is in top 1 (weak correctness test).
            self.assertIn('African_elephant', names)
        else:
            # Test correct label is in top 2 (weak correctness test).
            top_indices = preds[0].argsort()[-2:][::-1]
            self.assertIn(3674, top_indices)  # Fails on RepLKNet31L384K21


if __name__ == '__main__':
    tf.test.main()
