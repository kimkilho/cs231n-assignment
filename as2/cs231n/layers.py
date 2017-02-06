import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  num_examples = x.shape[0]
  x_vector = np.reshape(x, (num_examples, -1))  # (N, D)
  out = x_vector.dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  num_examples = x.shape[0]
  x_vector = np.reshape(x, (num_examples, -1)) # (N, D)
  dx = dout.dot(w.T)
  dx = np.reshape(dx, x.shape)
  dw = x_vector.T.dot(dout)
  db = np.sum(dout, axis=0)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0, x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = dout
  dx[x <= 0] = 0
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    normalized_x = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * normalized_x + beta
    cache = (x, sample_mean, sample_var, eps, normalized_x, gamma)

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

  elif mode == 'test':
    normalized_x = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * normalized_x + beta

  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  N = dout.shape[0]
  x, mean, var, eps, normalized_x, gamma = cache
  # mean: (D,)
  # var: (D,)
  
  dx = np.zeros_like(x)
  dmean = np.zeros_like(mean)
  dvar = np.zeros_like(var)

  dbeta = np.sum(dout, axis=0)  # (D,)
  dgamma = np.sum(dout * normalized_x, axis=0)  # (D,)
  dnormalized_x = dout * gamma  # (N, D)
  dx_m_mean = dnormalized_x / np.sqrt(var+eps)  # (N, D)
  dmean += -np.sum(dx_m_mean, axis=0) # (D,)
  dx += dx_m_mean

  d1_over_sqrt_var_p_eps = np.sum(dnormalized_x * (x - mean), axis=0) # (D,)
  dsqrt_var_p_eps = d1_over_sqrt_var_p_eps * (-1 / (var+eps)) # (D,)
  dvar_p_eps = dsqrt_var_p_eps / (2 * np.sqrt(var+eps)) # (D,)
  dvar = dvar_p_eps
  dvar_t_1_over_N = dvar / N
  dx_m_mean_squared = dvar_t_1_over_N * np.ones_like(x) # (N, D)
  dx_m_mean = dx_m_mean_squared * 2 * (x - mean)  # (N, D)
  dx += dx_m_mean
  dmean += -np.sum(dx_m_mean, axis=0) # (D,)
  dmean_t_1_over_N = dmean / N
  dx += dmean_t_1_over_N * np.ones_like(x)  # (N, D)

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  N = dout.shape[0]
  x, mean, var, eps, normalized_x, gamma = cache
  # mean: (D,)
  # var: (D,)
  
  dbeta = np.sum(dout, axis=0)  # (D,)
  dgamma = np.sum(dout * normalized_x, axis=0)  # (D,)
  dnormalized_x = dout * gamma  # (N, D)

  dvar = np.sum(dnormalized_x * (x - mean) * (-0.5*(var+eps)**(-1.5)), axis=0)
  dmean = np.sum(dnormalized_x * (-1/np.sqrt(var+eps)), axis=0) + \
          dvar * np.mean(-2*(x - mean), axis=0)

  dx = dnormalized_x/np.sqrt(var + eps) + \
       dvar * 2/N*(x - mean) + dmean/N
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = dout * mask 
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  stride = conv_param["stride"]
  P = conv_param["pad"]
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  out = np.zeros((N, F, 1+(H+2*P-HH)/stride, 1+(W+2*P-WW)/stride))
  x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), "constant", constant_values=0)
  for f in range(F):
    for c in range(C):
      for i in range(0, 1+(H+2*P-HH), stride):
        for j in range(0, 1+(W+2*P-WW), stride):
          out[:, f, i/stride, j/stride] += np.sum(x_padded[:,[c],i:i+HH,j:j+WW]*w[[f],[c],:,:], axis=(1, 2, 3))
          # (N, 1, 1, 1) += np.sum((N, 1, HH, WW) * (1, 1, HH, WW), axis=(1, 2, 3))
    out[:, f, :, :] += b[f]

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  stride = conv_param["stride"]
  P = conv_param["pad"]
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  dx = np.zeros_like(x) # (N, C, H, W)
  dw = np.zeros_like(w) # (F, C, HH, WW)
  db = np.zeros_like(b) # (F,)
  x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), "constant", constant_values=0)
  dx_padded = np.pad(dx, ((0, 0), (0, 0), (P, P), (P, P)), "constant", constant_values=0)
  for f in range(F):
    for c in range(C):
      for i in range(0, 1+(H+2*P-HH), stride):
        for j in range(0, 1+(W+2*P-WW), stride):
          dx_padded[:,[c],i:i+HH,j:j+WW] += \
            dout[:, f,np.newaxis, i/stride,np.newaxis, j/stride,np.newaxis] * w[f,np.newaxis, c,np.newaxis, :, :]
          # (N, 1, HH, WW) = (N, 1, 1, 1) * (1, 1, HH, WW)
          dw[f,c,:,:] += np.sum(dout[:, f,np.newaxis, i/stride,np.newaxis, j/stride,np.newaxis] * \
                                x_padded[:, c,np.newaxis, i:i+HH, j:j+WW], axis=(0, 1))
          # (HH, WW) = np.sum((N, 1, 1, 1), axis=0) * (N, 1, HH, WW)
    db[f] = np.sum(dout[:, f, :, :])
  dx = dx_padded[:, :, P:-P, P:-P]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  HH = pool_param["pool_height"]
  WW = pool_param["pool_width"]
  stride = pool_param["stride"]
  N, C, H, W = x.shape
  out = np.zeros((N, C, (H-HH)/stride+1, (W-WW)/stride+1))
  for c in range(C):
    for i in range(0, 1+(H-HH), stride):
      for j in range(0, 1+(W-WW), stride):
        out[:, c, i/stride, j/stride] += np.max(x[:, c,np.newaxis, i:i+HH, j:j+WW], axis=(1, 2, 3))
        # (N, 1, 1, 1) += np.max((N, 1, HH, WW), axis=(1, 2, 3))
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  HH = pool_param["pool_height"]
  WW = pool_param["pool_width"]
  stride = pool_param["stride"]
  N, C, H, W = x.shape
  dx = np.zeros_like(x) # (N, C, H, W)
  # dout: (N, C, (H-HH)/stride+1, (W-WW)/stride+1)
  for c in range(C):
    for i in range(0, 1+(H-HH), stride):
      for j in range(0, 1+(W-WW), stride):
        x_vectorized = np.reshape(x[:, c,np.newaxis, i:i+HH, j:j+WW], (N, -1))  # (N, HH*WW)
        argmax_vector = np.argmax(x_vectorized, axis=1)  # (N,)
        argmax_i, argmax_j = argmax_vector / HH, argmax_vector % HH  # (N,)
        dx[np.arange(dx.shape[0]), c, i+argmax_i, j+argmax_j] += dout[np.arange(dout.shape[0]), c, i/stride, j/stride]
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape

  x_C_dim_vect = x.transpose((1, 0, 2, 3)).reshape((C, -1)).transpose((1, 0))  # (N*H*W, C)
  out_C_dim_vect, cache = batchnorm_forward(x_C_dim_vect, gamma, beta, bn_param)
  out = out_C_dim_vect.transpose((1, 0)).reshape((C, N, H, W)).transpose((1, 0, 2, 3))

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  N, C, H, W = dout.shape

  dout_C_dim_vect = dout.transpose((1, 0, 2, 3)).reshape((C, -1)).transpose((1, 0))  # (N*H*W, C)
  dx_C_dim_vect, dgamma, dbeta = batchnorm_backward(dout_C_dim_vect, cache)
  dx = dx_C_dim_vect.transpose((1, 0)).reshape((C, N, H, W)).transpose((1, 0, 2, 3))

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
