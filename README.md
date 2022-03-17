# tfreplknet

Keras (TensorFlow v2) reimplementation of **Re-parameterized Large Kernel Network (RepLKNet)** model.

Based on [Official Pytorch implementation](https://github.com/DingXiaoH/RepLKNet-pytorch).

Supports variable-shape inference.

## Examples

Default usage (without preprocessing):

```python
from tfreplknet import RepLKNet31B224K1  # + 4 other variants and input preprocessing

model = RepLKNet31B224K1()  # by default will download imagenet{1k, 21k}-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from keras import layers, models
from tfreplknet import RepLKNet31B224K1, preprocess_input

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = layers.Lambda(preprocess_input)(inputs)
outputs = RepLKNet31B224K1(include_top=False)(outputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(...)
model.fit(...)
```

## Evaluation

For correctness, `RepLKNet31B224K1` and `RepLKNet31B384K1` models (original and ported) tested
with [ImageNet-v2 test set](https://www.tensorflow.org/datasets/catalog/imagenet_v2).

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tfreplknet import RepLKNet31B224K1, preprocess_input

def _prepare(example):
    image = tf.image.resize(example['image'], (256, 256), method=tf.image.ResizeMethod.BICUBIC, antialias=False)
    image = tf.image.central_crop(image, 0.875)
    image = preprocess_input(image)
    
    return image, example['label']
    
imagenet2 = tfds.load('imagenet_v2', split='test', shuffle_files=True)
imagenet2 = imagenet2.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
imagenet2 = imagenet2.batch(8)

model = RepLKNet31B224K1()
model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy', 'sparse_top_k_categorical_accuracy'])
history = model.evaluate(imagenet2)

print(history)
```

| name | original acc@1 | ported acc@1 | original acc@5 | ported acc@5 |
| :---: | :---: | :---: | :---: | :---: |
| RepLKNet31B 224 1K | ? | ? | ? | ? |
| RepLKNet31B 384 1K | ? | ? | ? | ? |

## Citation

```
@article{2022arXiv220306717D,
  title={Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs},
  author={{Ding}, Xiaohan and {Zhang}, Xiangyu and {Zhou}, Yizhuang and {Han}, Jungong and {Ding}, Guiguang and {Sun}, Jian},
  journal={arXiv preprint arXiv:2203.06717},
  year={2022}
}
