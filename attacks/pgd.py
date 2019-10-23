"""
The Projected Gradient Descent attack.
Code source: https://github.com/tensorflow/cleverhans/tree/master/cleverhans/future/tf2/attacks
"""

import numpy as np
import tensorflow as tf

from attacks.fgsm import FGSM

def clip_eta(eta, norm, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param norm: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.norm norm ball
  if norm not in [np.inf, 0, 2]:
    raise ValueError('norm must be np.inf, 0, or 2.')
  axis = list(range(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if norm == np.inf:
    eta = tf.clip_by_value(eta, -eps, eps)
  else:
    if norm == 0:
      raise NotImplementedError("")
      # This is not the correct way to project on the L1 norm ball:
      # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta), reduc_ind, keepdims=True))
    elif norm == 2:
      # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
      norm = tf.sqrt(
        tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
    factor = tf.minimum(1., tf.math.divide(eps, norm))
    eta = eta * factor
  return eta



def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm,
                               clip_min=None, clip_max=None, y=None, targeted=False,
                               rand_init=None, rand_minmax=0.3, sanity_checks=True):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param eps_iter: step size for each attack iteration
  :param nb_iter: Number of attack iterations.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """

  assert eps_iter <= eps, (eps_iter, eps)
  if norm == 0:
    raise NotImplementedError("It's not clear that FGSM is a good inner loop"
                              " step for PGD when norm=0, because norm=1 FGSM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong norm=1 PGD "
                              "before enabling this feature.")
  if norm not in [np.inf, 2]:
    raise ValueError("Norm order must be either np.inf or 2.")

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(tf.math.greater_equal(x, clip_min))

  if clip_max is not None:
    asserts.append(tf.math.less_equal(x, clip_max))

  # Initialize loop variables
  if rand_init:
    rand_minmax = eps
    eta = tf.random.uniform(x.shape, -rand_minmax, rand_minmax)
  else:
    eta = tf.zeros_like(x)

  # Clip eta
  eta = clip_eta(eta, norm, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    y = tf.argmax(model_fn(x), 1)

  i = 0
  while i < nb_iter:
    adv_x = FGSM(model_fn, adv_x, eps_iter, norm, clip_min=clip_min,
                                 clip_max=clip_max, y=y, targeted=targeted)

    # Clipping perturbation eta to norm norm ball
    eta = adv_x - x
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta

    # Redo the clipping.
    # FGSM already did it, but subtracting and re-adding eta can add some
    # small numerical error.
    if clip_min is not None or clip_max is not None:
      adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    i += 1

  asserts.append(eps_iter <= eps)
  if norm == np.inf and clip_min is not None:
    # TODO necessary to cast to x.dtype?
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x
