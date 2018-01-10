import tensorflow as tf
import numpy as np

def softmax_cross_entropy_with_self_paced(
    logits, onehot_labels, miscls_cost, weights=1.0, label_smoothing=0, scope=None):
  """Creates a cross-entropy loss with self-paced-learning using tf.nn.softmax_cross_entropy_with_logits.

  Self-paced Learning for Imbalanced Data(https://link.springer.com/chapter/10.1007/978-3-662-49381-6_54)

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] one-hot-encoded labels.
    miscls_cost: [num_classes] misclassification cost for cost sensitive self-paced learning
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the mean loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.
  """
  with tf.name_scope(scope, "softmax_cross_entropy_loss",
                      [logits, onehot_labels, weights]) as scope:
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    onehot_labels = tf.cast(onehot_labels, logits.dtype)

    if label_smoothing > 0:
      num_classes = tf.cast(
          tf.array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,
                                                  logits=logits,
                                                  name="xentropy")

    mask_cls_0 = tf.cast(onehot_labels[0], tf.bool)
    mask_cls_1 = tf.cast(onehot_labels[1], tf.bool)

    losses_cls_0 = tf.boolean_mask(losses, mask_cls_0)
    losses_cls_1 = tf.boolean_mask(losses, mask_cls_1)

    mask_miscls_0 = tf.less(losses_cls_0, tf.ones_like(losses_cls_0) * miscls_cost[0])
    mask_miscls_1 = tf.less(losses_cls_1, tf.ones_like(losses_cls_1) * miscls_cost[1])

    sp_losses_0 = tf.boolean_mask(losses_cls_0, mask_miscls_0) * miscls_cost[0]
    sp_losses_1 = tf.boolean_mask(losses_cls_1, mask_miscls_1) * miscls_cost[1]
    
    losses = tf.concat([sp_losses_0, sp_losses_1], 0)

    return tf.losses.compute_weighted_loss(losses, weights, scope=scope)


def mean_squared_error_with_self_paced(
    logits, onehot_labels, miscls_cost, weights=1.0, label_smoothing=0, scope=None):
  """Creates a mean squared error with self-paced learning using tf.squared_difference.

  Self-paced Learning for Imbalanced Data(https://link.springer.com/chapter/10.1007/978-3-662-49381-6_54)

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] one-hot-encoded labels.
    miscls_cost: [num_classes] misclassification cost for cost sensitive self-paced learning
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the mean loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.
  """
  with tf.name_scope(scope, "mean_squared_error_with_self_paced",
                      [logits, onehot_labels, weights]) as scope:
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    onehot_labels = tf.cast(onehot_labels, logits.dtype)

    if label_smoothing > 0:
      num_classes = tf.cast(
          tf.array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    softmax_logits = tf.nn.softmax(logits)
    losses = tf.squared_difference(onehot_labels, softmax_logits)
    losses = tf.reduce_mean(losses, 1)

    mask_cls_0 = tf.cast(onehot_labels[0], tf.bool)
    mask_cls_1 = tf.cast(onehot_labels[1], tf.bool)

    losses_cls_0 = tf.boolean_mask(losses, mask_cls_0)
    losses_cls_1 = tf.boolean_mask(losses, mask_cls_1)

    mask_miscls_0 = tf.less(losses_cls_0, tf.ones_like(losses_cls_0) * miscls_cost[0])
    mask_miscls_1 = tf.less(losses_cls_1, tf.ones_like(losses_cls_1) * miscls_cost[1])

    sp_losses_0 = tf.boolean_mask(losses_cls_0, mask_miscls_0) * miscls_cost[0]
    sp_losses_1 = tf.boolean_mask(losses_cls_1, mask_miscls_1) * miscls_cost[1]
    
    losses = tf.concat([sp_losses_0, sp_losses_1], 0)

    return tf.losses.compute_weighted_loss(losses, weights, scope=scope)

def configure_misclassification_cost(global_step, initial_cost=np.array([1, 0.01]), step_size=0.01):
  """Configures the misclassification cost.

  Args:
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the misclassification_cost.

  Raises:
    ValueError: if
  """
  if global_step is None:
    raise ValueError("global_step is required for configure_misclassification_cost.")
  
  mu = (global_step//300) * step_size * 100
  misclassification_cost = initial_cost * (1 + mu)

  return tf.constant(misclassification_cost, name='misclassification_cost')
