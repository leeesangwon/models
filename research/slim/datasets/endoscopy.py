# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'classification_data_of_3_images_%s_224.tfrecord'
SPLITS_TO_SIZES = {'train': 490, 'validation': 122}
_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 1',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  
  if not file_pattern:
    file_pattern = _FILE_PATTERN
    splits_to_sizes = SPLITS_TO_SIZES
  else:
    if file_pattern == 'classification_data_of_3_images_0_%s_224_20180117.tfrecord':
      splits_to_sizes = {'train': 484, 'test': 135}
    elif file_pattern == 'classification_data_of_3_images_1_%s_224_20180117.tfrecord':
      splits_to_sizes = {'train': 492, 'test': 127}
    elif file_pattern == 'classification_data_of_3_images_2_%s_224_20180117.tfrecord':
      splits_to_sizes = {'train': 483, 'test': 136}
    elif file_pattern == 'classification_data_of_3_images_3_%s_224_20180117.tfrecord':
      splits_to_sizes = {'train': 490, 'test': 129}
    elif file_pattern == 'classification_data_of_3_images_4_%s_224_20180117.tfrecord':
      splits_to_sizes = {'train': 527, 'test': 92}
    else:
      splits_to_sizes = SPLITS_TO_SIZES
    
  
  if split_name not in splits_to_sizes:
    raise ValueError('split name %s was not recognized.' % split_name)

  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': Image9ch(shape=[224, 224, 9]),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)


class Image9ch(slim.tfexample_decoder.ItemHandler):
  def __init__(self,
               image_key=None,
               format_key=None,
               shape=None,
               channels=9,
               dtype=tf.float32,
               repeated=False):
    """Initializes the image.

    Args:
      image_key: the name of the TF-Example feature in which the encoded image
        is stored.
      format_key: the name of the TF-Example feature in which the image format
        is stored.
      shape: the output shape of the image as 1-D `Tensor`
        [height, width, channels]. If provided, the image is reshaped
        accordingly. If left as None, no reshaping is done. A shape should
        be supplied only if all the stored images have the same shape.
      channels: the number of channels in the image.
      dtype: images will be decoded at this bit depth. Different formats
        support different bit depths.
          See tf.image.decode_image,
              tf.decode_raw,
      repeated: if False, decodes a single image. If True, decodes a
        variable number of image strings from a 1D tensor of strings.
    """
    if not image_key:
      image_key = 'image/encoded'
    if not format_key:
      format_key = 'image/format'

    super(Image9ch, self).__init__([image_key, format_key])
    self._image_key = image_key
    self._format_key = format_key
    self._shape = shape
    self._channels = channels
    self._dtype = dtype
    self._repeated = repeated

  def tensors_to_item(self, keys_to_tensors):
    """See base class."""
    image_buffer = keys_to_tensors[self._image_key]
    image_format = keys_to_tensors[self._format_key]

    if self._repeated:
      return tf.map_fn(lambda x: self._decode(x, image_format),
                                   image_buffer, dtype=self._dtype)
    else:
      return self._decode(image_buffer, image_format)

  def _decode(self, image_buffer, image_format):
    """Decodes the image buffer.

    Args:
      image_buffer: The tensor representing the encoded image tensor.
      image_format: The image format for the image in `image_buffer`. If image
        format is `raw`, all images are expected to be in this format, otherwise
        this op can decode a mix of `jpg` and `png` formats.

    Returns:
      A tensor that represents decoded image of self._shape, or
      (?, ?, self._channels) if self._shape is not specified.
    """

    image = tf.decode_raw(image_buffer, out_type=self._dtype)

    if self._shape is not None:
      image = tf.reshape(image, self._shape)

    return image