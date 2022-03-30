# tfreplknet

Keras (TensorFlow v2) reimplementation of **Re-parameterized Large Kernel Network (RepLKNet)** model.

Based on [Official Pytorch implementation](https://github.com/DingXiaoH/RepLKNet-pytorch).

Supports variable-shape inference.

## Installation

```bash
pip install tfreplknet
```

## Available models and weights

| Model name | Pretrain size | Preprocessing function | Description |
| :---: | :---: | :---: | :---: |
| RepLKNet | - | - | General RepLKNet architecture |
| RepLKNetB | - | - | Base model size preset |
| RepLKNetL | - | - | Large model size preset |
| RepLKNetXL | - | - | Extra large model size preset |
| RepLKNetB224In1k | 224 | preprocess_input_bl | Base model with weighs pretrained on ImageNet 21k and finetuned to 1k |
| RepLKNetB224In21k | 224 | preprocess_input_bl | Base model with weighs pretrained on ImageNet 21k |
| RepLKNetB384In1k | 384 | preprocess_input_bl | Base model with weighs pretrained on ImageNet 21k and finetuned to 1k |
| RepLKNetL384In1k | 384 | preprocess_input_bl | Large model with weighs pretrained on ImageNet 21k and finetuned to 1k |
| RepLKNetL384In21k | 384 | preprocess_input_bl | Large model with weighs pretrained on ImageNet 21k |
| RepLKNetXL320In1k | 320 | preprocess_input_xl | Extra large model with weighs pretrained on MegData-73M and finetuned to 1k |
| RepLKNetXL320In21k | 320 | preprocess_input_xl | Extra large model with weighs pretrained on MegData-73M (21k head) |


## Examples

Default usage (without preprocessing):

```python
from tfreplknet import RepLKNetB224In1k  # + 4 other variants and input preprocessing

model = RepLKNetB224In1k()  # by default will download imagenet{1k, 21k}-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from keras import layers, models
from tfreplknet import RepLKNetB224In1k, preprocess_input_bl

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = layers.Lambda(preprocess_input_bl)(inputs)
outputs = RepLKNetB224In1k(include_top=False)(outputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(...)
model.fit(...)
```

## Evaluation

For correctness, `RepLKNetB224In1k` and `RepLKNetB384In1k` models (original and ported) tested
with [ImageNet-v2 test set](https://www.tensorflow.org/datasets/catalog/imagenet_v2).

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tfreplknet import RepLKNetB224In1k, RepLKNetB384In1k, preprocess_input_bl

def _prepare(example):
    # For RepLKNetB224In1k
    image = tf.image.resize(example['image'], (256, 256), method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.central_crop(image, 0.875)
    
    # For RepLKNetB384In1k
    # image = tf.image.resize(example['image'], (438, 438), method=tf.image.ResizeMethod.BICUBIC)
    # image = tf.image.central_crop(image, 0.877)
    
    image = preprocess_input_bl(image)
    
    return image, example['label']
    
imagenet2 = tfds.load('imagenet_v2', split='test', shuffle_files=True)
imagenet2 = imagenet2.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
imagenet2 = imagenet2.batch(8)

model = RepLKNetB224In1k()
model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy', 'sparse_top_k_categorical_accuracy'])
history = model.evaluate(imagenet2)

print(history)
```

| name | original acc@1 | ported acc@1 | original acc@5 | ported acc@5 |
| :---: | :---: | :---: | :---: | :---: |
| RepLKNetB 224 1K | 75.29 | 75.13 | 92.60 | 92.88 |
| RepLKNetB 384 1K | 72.77 | 76.46 | 89.91 | 93.37 |

## Citation

```
@article{2022arXiv220306717D,
  title={Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs},
  author={{Ding}, Xiaohan and {Zhang}, Xiangyu and {Zhou}, Yizhuang and {Han}, Jungong and {Ding}, Guiguang and {Sun}, Jian},
  journal={arXiv preprint arXiv:2203.06717},
  year={2022}
}
