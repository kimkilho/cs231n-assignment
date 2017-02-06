import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    self.params["W1"] = np.random.normal(loc=0.0, scale=weight_scale, 
                                         size=(input_dim, hidden_dim))
    self.params["b1"] = np.zeros((hidden_dim))
    self.params["W2"] = np.random.normal(loc=0.0, scale=weight_scale,
                                         size=(hidden_dim, num_classes))
    self.params["b2"] = np.zeros((num_classes))


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]

    fc1, fc1_cache = affine_forward(X, W1, b1)
    relu1, relu1_cache = relu_forward(fc1)
    scores, scores_cache = affine_forward(relu1, W2, b2)

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    # The architecure should be affine - relu - affine - softmax.
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
    drelu1, dW2, db2 = affine_backward(dscores, scores_cache)
    dW2 += self.reg * W2
    grads["W2"] = dW2; grads["b2"] = db2
    dfc1 = relu_backward(drelu1, relu1_cache)
    dX, dW1, db1 = affine_backward(dfc1, fc1_cache)
    dW1 += self.reg * W1
    grads["W1"] = dW1; grads["b1"] = db1

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    layer_dims = [input_dim] + hidden_dims + [num_classes]
    for i in range(0, self.num_layers):
      self.params["W%d" % (i+1)] = \
        np.random.normal(loc=0.0, scale=weight_scale, 
                         size=(layer_dims[i], layer_dims[i+1]))
      self.params["b%d" % (i+1)] = np.zeros((layer_dims[i+1]))

      if self.use_batchnorm:
        if not i == self.num_layers - 1:
          self.params["gamma%d" % (i+1)] = np.ones((layer_dims[i+1]))
          self.params["beta%d" % (i+1)] = np.zeros((layer_dims[i+1]))

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    out_dict = {}
    relu = X
    for i in range(0, self.num_layers):
      W, b = self.params["W%d" % (i+1)], self.params["b%d" % (i+1)]
      fc, fc_cache = affine_forward(relu, W, b)
      out_dict["fc%d_cache" % (i+1)] = fc_cache
      if i == self.num_layers - 1: 
        continue
      # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
      if self.use_batchnorm:
        gamma, beta = self.params["gamma%d" % (i+1)], self.params["beta%d" % (i+1)]
        bn, bn_cache = batchnorm_forward(fc, gamma, beta, self.bn_params[i])
        out_dict["bn%d_cache" % (i+1)] = bn_cache
        relu, relu_cache = relu_forward(bn)
      else:
        relu, relu_cache = relu_forward(fc)
      if self.use_dropout:
        relu, dropout_cache = dropout_forward(relu, self.dropout_param)
        out_dict["dropout%d_cache" % (i+1)] = dropout_cache
      out_dict["relu%d_cache" % (i+1)] = relu_cache
    scores, scores_cache = fc, fc_cache

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores, y)
    for i in range(0, self.num_layers):
      W = self.params["W%d" % (i+1)]
      loss += 0.5 * self.reg * np.sum(W * W)

    drelu, dW, db = affine_backward(dscores, scores_cache)
    # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    for i in range(self.num_layers-1, -1, -1):
      W, b = self.params["W%d" % (i+1)], self.params["b%d" % (i+1)]
      if not i == self.num_layers - 1:
        if self.use_dropout:
          dropout_cache = out_dict["dropout%d_cache" % (i+1)]
          drelu = dropout_backward(drelu, dropout_cache)
        relu_cache = out_dict["relu%d_cache" % (i+1)]
        dfc = relu_backward(drelu, relu_cache)
        if self.use_batchnorm:
          dbn = dfc
          bn_cache = out_dict["bn%d_cache" % (i+1)]
          dfc, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
          grads["gamma%d" % (i+1)] = dgamma
          grads["beta%d" % (i+1)] = dbeta
        fc_cache = out_dict["fc%d_cache" % (i+1)]
        drelu, dW, db = affine_backward(dfc, fc_cache)
      dW += self.reg * W
      grads["W%d" % (i+1)], grads["b%d" % (i+1)] = dW, db

    return loss, grads
