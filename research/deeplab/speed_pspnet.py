# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Segmentation speed measurement on a given set of images.

See model.py for more details and usage.
"""
import math
import os.path
import time
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model_pspnet as model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('speed_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for measuring inference time.

flags.DEFINE_integer('speed_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('speed_crop_size', [513, 513],
                           'Crop size [height, width] for measurment.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('speed_split', 'val',
                    'Which split of the dataset used for measuring inference time')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


def _process_batch(sess, semantic_predictions):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    semantic_predictions: One batch of semantic segmentation predictions.

  Returns:
    Mean inference time in batch.
  """
  start = time.time()
  semantic_predictions = sess.run(semantic_predictions)
  inference_time = time.time() - start
  num_image = semantic_predictions.shape[0]
  inference_time /= num_image
  return inference_time


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.speed_split, dataset_dir=FLAGS.dataset_dir)

  tf.logging.info('Measuring inference time on %s set', FLAGS.speed_split)

  g = tf.Graph()
  with g.as_default():
    samples = input_generator.get(dataset,
                                  FLAGS.speed_crop_size,
                                  FLAGS.speed_batch_size,
                                  min_resize_value=FLAGS.min_resize_value,
                                  max_resize_value=FLAGS.max_resize_value,
                                  resize_factor=FLAGS.resize_factor,
                                  dataset_split=FLAGS.speed_split,
                                  is_training=False,
                                  model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.speed_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.speed_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we reisze the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.speed_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)
    num_batches = int(math.ceil(
        dataset.num_samples / float(FLAGS.speed_batch_size)))

    last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
        FLAGS.checkpoint_dir, None)
        
    tf.logging.info('Measuring inference time with model %s', last_checkpoint)

    with sv.managed_session(FLAGS.master,
                            start_standard_services=False) as sess:
      sv.start_queue_runners(sess)
      sv.saver.restore(sess, last_checkpoint)

      image_id_offset = 0
      inference_times = []
      for batch in range(num_batches):
        inference_time = _process_batch(sess=sess, semantic_predictions=predictions)
        tf.logging.info('Inference time of batch %3d / %d\t%5.2f ms', batch + 1, num_batches, inference_time * 1000)
        inference_times.append(inference_time)
        image_id_offset += FLAGS.speed_batch_size

    mean_inference_time = sum(inference_times) / float(len(inference_times))
    tf.logging.info('Mean inference time\t%5.2f ms', mean_inference_time * 1000)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('speed_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
