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
r"""Provides DeepLab model definition and helper functions.

DeepLab is a deep learning system for semantic image segmentation with
the following features:

(1) Atrous convolution to explicitly control the resolution at which
feature responses are computed within Deep Convolutional Neural Networks.

(2) Atrous spatial pyramid pooling (ASPP) to robustly segment objects at
multiple scales with filters at multiple sampling rates and effective
fields-of-views.

(3) ASPP module augmented with image-level feature and batch normalization.

(4) A simple yet effective decoder module to recover the object boundaries.

See the following papers for more details:

"Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation"
Liang-Chieh Chen, Yukun Zhu, George Papandmodelreou, Florian Schroff, Hartwig Adam.
(https://arxiv.org/abs/1802.02611)

"Rethinking Atrous Convolution for Semantic Image Segmentation,"
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
(https://arxiv.org/abs/1706.05587)

"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
Atrous Convolution, and Fully Connected CRFs",
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L Yuille (* equal contribution)
(https://arxiv.org/abs/1606.00915)

"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected
CRFs"
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L. Yuille (* equal contribution)
(https://arxiv.org/abs/1412.7062)
"""
import tensorflow as tf
from deeplab.core import feature_extractor

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
PPM_SCOPE = 'ppm'
CONCAT_PROJECTION_SCOPE = 'concat_projection' 


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        PPM_SCOPE,
        CONCAT_PROJECTION_SCOPE,
    ]


def predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    eval_scales: The scales to resize images for evaluation.
    add_flipped_images: Add flipped images for evaluation or not.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
      outputs_to_scales_to_logits = multi_scale_logits(
          images,
          model_options=model_options,
          image_pyramid=[image_scale],
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = multi_scale_logits(
            tf.reverse_v2(images, [2]),
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = tf.image.resize_bilinear(
            tf.reverse_v2(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            align_corners=True)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    # Compute average prediction across different scales and flipped images.
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = tf.argmax(predictions, 3)

  return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = tf.image.resize_bilinear(
        scales_to_logits[MERGED_LOGITS_SCOPE],
        tf.shape(images)[1:3],
        align_corners=True)
    predictions[output] = tf.argmax(logits, 3)

  return predictions


def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
  """Gets the logits for multi-scale inputs.

  The returned logits are all downsampled (due to max-pooling layers)
  for both training and evaluation.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.
    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
      semantic prediction) to a dictionary of multi-scale logits names to
      logits. For each output_type, the dictionary has keys which
      correspond to the scales and values which correspond to the logits.
      For example, if `scales` equals [1.0, 1.5], then the keys would
      include 'merged_logits', 'logits_1.00' and 'logits_1.50'.

  Raises:
    ValueError: If model_options doesn't specify crop_size and its
      add_image_level_feature = True, since add_image_level_feature requires
      crop_size information.
  """
  # Setup default values.
  if not image_pyramid:
    image_pyramid = [1.0]
  if model_options.crop_size is None and model_options.add_image_level_feature:
    raise ValueError(
        'Crop size must be specified for using image-level feature.')
  crop_height = (
      model_options.crop_size[0]
      if model_options.crop_size else tf.shape(images)[1])
  crop_width = (
      model_options.crop_size[1]
      if model_options.crop_size else tf.shape(images)[2])

  # Compute the height, width for the output logits.
  logits_output_stride = (
      model_options.decoder_output_stride or model_options.output_stride)

  logits_height = scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)

  # Compute the logits for each scale in the image pyramid.
  outputs_to_scales_to_logits = {
      k: {}
      for k in model_options.outputs_to_num_classes
  }

  for image_scale in image_pyramid:
    if image_scale != 1.0:
      scaled_height = scale_dimension(crop_height, image_scale)
      scaled_width = scale_dimension(crop_width, image_scale)
      scaled_crop_size = [scaled_height, scaled_width]
      scaled_images = tf.image.resize_bilinear(
          images, scaled_crop_size, align_corners=True)
      if model_options.crop_size:
        scaled_images.set_shape([None, scaled_height, scaled_width, 3])
    else:
      scaled_crop_size = model_options.crop_size
      scaled_images = images

    updated_options = model_options._replace(crop_size=scaled_crop_size)
    outputs_to_logits = _get_logits(
        scaled_images,
        updated_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    # Resize the logits to have the same dimension before merging.
    for output in sorted(outputs_to_logits):
      outputs_to_logits[output] = tf.image.resize_bilinear(
          outputs_to_logits[output], [logits_height, logits_width],
          align_corners=True)

    # Return when only one input scale.
    if len(image_pyramid) == 1:
      for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_scales_to_logits[output][
            MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(model_options.outputs_to_num_classes):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi-scale inputs.
  for output in sorted(model_options.outputs_to_num_classes):
    # Concatenate the multi-scale logits for each output type.
    all_logits = [
        tf.expand_dims(logits, axis=4)
        for logits in outputs_to_scales_to_logits[output].values()
    ]
    all_logits = tf.concat(all_logits, 4)
    merge_fn = (
        tf.reduce_max
        if model_options.merge_method == 'max' else tf.reduce_mean)
    outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(
        all_logits, axis=4)

  return outputs_to_scales_to_logits


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=model_options.output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if not model_options.aspp_with_batch_norm:
    return features, end_points
  else:
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        branch_logits = []

        # Pyramid pooling module
        if model_options.atrous_rates:
          depth = int(320 / len(model_options.atrous_rates)) # num of mobilenet_v2 output channel = 320
          feature_height = scale_dimension(model_options.crop_size[0],
                                        1. / model_options.output_stride)
          feature_width = scale_dimension(model_options.crop_size[1],
                                       1. / model_options.output_stride)
          with tf.variable_scope(PPM_SCOPE):
            for i, pool in enumerate(model_options.atrous_rates, 1):
              with tf.variable_scope(IMAGE_POOLING_SCOPE + str(i)) as scope:
                pool_height = feature_height / pool
                pool_width = feature_height / pool
                ppm_feature = slim.avg_pool2d(
                    features, [pool_height, pool_width], [pool_height, pool_width],
                    padding='VALID')
                ppm_feature = slim.conv2d(
                    ppm_feature, depth, 1, scope=scope)
                ppm_feature = tf.image.resize_bilinear(
                    ppm_feature, [feature_height, feature_width], align_corners=True)
                ppm_feature.set_shape([None, feature_height, feature_width, depth])
                branch_logits.append(ppm_feature)

        # Input feature
        branch_logits.append(features)

        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = slim.conv2d(
            concat_logits, depth, 3, scope=CONCAT_PROJECTION_SCOPE)
        concat_logits = slim.dropout(
            concat_logits,
            keep_prob=0.9,
            is_training=is_training,
            scope=CONCAT_PROJECTION_SCOPE + '_dropout')

        return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):
  """Gets the logits by atrous/image spatial pyramid pooling.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    outputs_to_logits: A map from output_type to logits.
  """
  features, end_points = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  outputs_to_logits = {}
  for output in sorted(model_options.outputs_to_num_classes):
    outputs_to_logits[output] = get_branch_logits(
        features,
        model_options.outputs_to_num_classes[output],
        model_options.atrous_rates,
        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
        kernel_size=model_options.logits_kernel_size,
        weight_decay=weight_decay,
        reuse=reuse,
        scope_suffix=output)

  return outputs_to_logits


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
  """Gets the logits from each model's branch.

  The underlying model is branched out in the last layer when atrous
  spatial pyramid pooling is employed, and all branches are sum-merged
  to form the final logits.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    num_classes: Number of classes to predict.
    atrous_rates: A list of atrous convolution rates for last layer.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    kernel_size: Kernel size for convolution.
    weight_decay: Weight decay for the model variables.
    reuse: Reuse model variables or not.
    scope_suffix: Scope suffix for the model variables.

  Returns:
    Merged logits with shape [batch, height, width, num_classes].

  Raises:
    ValueError: Upon invalid input kernel_size value.
  """
  # When using batch normalization with ASPP, ASPP has been applied before
  # in extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')
