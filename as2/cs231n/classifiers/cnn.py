import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    self.params["W1"] = np.random.normal(loc=0.0, scale=weight_scale,
                                         size=(num_filters, C, filter_size, filter_size))
    self.params["b1"] = np.zeros((num_filters))

    # NOTE: Suppose the feature map size after convolution remains the same
    pool_size = 2
    pool_stride = 2
    pooled_H = (H - pool_size)/pool_stride + 1
    pooled_W = (W - pool_size)/pool_stride + 1

    self.params["W2"] = np.random.normal(loc=0.0, scale=weight_scale,
                                         size=(num_filters*pooled_H*pooled_W, hidden_dim))
    self.params["b2"] = np.zeros((hidden_dim))

    self.params["W3"] = np.random.normal(loc=0.0, scale=weight_scale,
                                         size=(hidden_dim, num_classes))
    self.params["b3"] = np.zeros((num_classes))

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    conv1, conv1_cache = conv_forward_fast(X, W1, b1, conv_param)
    relu1, relu1_cache = relu_forward(conv1)
    pool1, pool1_cache = max_pool_forward_fast(relu1, pool_param)

    fc2, fc2_cache = affine_forward(pool1, W2, b2)
    relu2, relu2_cache = relu_forward(fc2)

    scores, scores_cache = affine_forward(relu2, W3, b3)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

    drelu2, dW3, db3 = affine_backward(dscores, scores_cache)
    dW3 += self.reg * W3
    grads["W3"] = dW3; grads["b3"] = db3

    dfc2 = relu_backward(drelu2, relu2_cache)
    dpool1, dW2, db2 = affine_backward(dfc2, fc2_cache)
    dW2 += self.reg * W2
    grads["W2"] = dW2; grads["b2"] = db2

    drelu1 = max_pool_backward_fast(dpool1, pool1_cache)
    dconv1 = relu_backward(drelu1, relu1_cache)
    dX, dW1, db1 = conv_backward_fast(dconv1, conv1_cache)
    dW1 += self.reg * W1
    grads["W1"] = dW1; grads["b1"] = db1
    
    return loss, grads
  

class ConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional layers, pooling layers,
  hidden layers, ReLU nonlinearities, and a softmax loss function.
  For a network with N conv layers and M hidden layers, the architecture will be:
  
  [[conv-relu]xP-2x2 max pool]xN - [affine-relu]xM - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_conv_layers=2, num_filters=[32, 32], filter_size=3,
               hidden_dims=[100], num_classes=10, dropout=0, use_batchnorm=False, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data.
    - num_conv_layers: Number of convolutional layers in each convolutional module.
    - num_filters: A list of integers giving the number of filters to use 
      in each convolutional module.
    - filter_size: Size of filters to use in the convolutional module(conv-relu-conv-relu).
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """

    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_conv_modules = len(num_filters)
    self.num_conv_layers = num_conv_layers
    self.num_hidden_layers = 1 + len(hidden_dims)
    self.num_total_modules = self.num_conv_modules + self.num_hidden_layers
    self.num_total_layers = self.num_conv_modules*self.num_conv_layers + self.num_hidden_layers
    self.dtype = dtype
    self.params = {}
    
    C, H, W = input_dim
    for i in range(0, self.num_conv_modules):
      for j in range(0, self.num_conv_layers):
        if i == 0 and j == 0: prev_num_filters = C
        elif j == 0:           prev_num_filters = num_filters[i-1]
        else:                  prev_num_filters = num_filters[i]
        self.params["W%d_%d" % (i+1, j+1)] = np.random.normal(loc=0.0, scale=weight_scale,
                                                           size=(num_filters[i], prev_num_filters, filter_size, filter_size))
        self.params["b%d_%d" % (i+1, j+1)] = np.zeros((num_filters[i]))

        if self.use_batchnorm:
          self.params["gamma%d_%d" % (i+1, j+1)] = np.ones((num_filters[i]))
          self.params["beta%d_%d" % (i+1, j+1)] = np.zeros((num_filters[i]))

    # NOTE: Suppose the feature map size after convolution remains the same
    pool_size = 2
    pool_stride = 2
    pooled_H = H
    pooled_W = W
    for i in range(self.num_conv_modules):
      pooled_H = (pooled_H - pool_size)/pool_stride + 1
      pooled_W = (pooled_W - pool_size)/pool_stride + 1

    hidden_dims = [num_filters[-1]*pooled_H*pooled_W] + hidden_dims + [num_classes]
    for i in range(self.num_conv_modules, self.num_conv_modules+self.num_hidden_layers):
      ii = i - self.num_conv_modules
      self.params["W%d" % (i+1)] = np.random.normal(loc=0.0, scale=weight_scale,
                                                    size=(hidden_dims[ii], hidden_dims[ii+1]))
      self.params["b%d" % (i+1)] = np.zeros((hidden_dims[ii+1]))

      if self.use_batchnorm:
        if not ii == self.num_hidden_layers - 1:
          self.params["gamma%d" % (i+1)] = np.ones((hidden_dims[ii+1]))
          self.params["beta%d" % (i+1)] = np.zeros((hidden_dims[ii+1]))

    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {"mode": "train", "p": dropout}
      if seed is not None:
        self.dropout_param["seed"] = seed

    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{"mode": "train"} for i in xrange(self.num_total_layers - 1)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the convolutional network.
    
    Input / output: Same API as FullyConnectedNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = "test" if y is None else "train"
    
    N, C, H, W = X.shape
    out_dict = {}
    relu = X
    for i in range(0, self.num_conv_modules):
      for j in range(0, self.num_conv_layers):
        # pass conv_param to the forward pass for the convolutional layer
        W, b = self.params["W%d_%d" % (i+1, j+1)], self.params["b%d_%d" % (i+1, j+1)]
        filter_size = W.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        conv, conv_cache = conv_forward_fast(relu, W, b, conv_param)
        out_dict["conv%d_%d_cache" % (i+1, j+1)] = conv_cache

        if self.use_batchnorm:
          gamma, beta = self.params["gamma%d_%d" % (i+1, j+1)], self.params["beta%d_%d" % (i+1, j+1)]
          bn, bn_cache = spatial_batchnorm_forward(conv, gamma, beta, self.bn_params[i*self.num_conv_modules+j])
          out_dict["bn%d_%d_cache" % (i+1, j+1)] = bn_cache
          relu, relu_cache = relu_forward(bn)
        else:
          relu, relu_cache = relu_forward(conv)
        if self.use_dropout:
          relu, dropout_cache = dropout_forward(relu, self.dropout_param)
          out_dict["dropout%d_%d_cache" % (i+1, j+1)] = dropout_cache
        out_dict["relu%d_%d_cache" % (i+1, j+1)] = relu_cache

      # pass pool_param to the forward pass for the max-pooling layer
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
      relu, pool_cache = max_pool_forward_fast(relu, pool_param)
      out_dict["pool%d_cache" % (i+1)] = pool_cache

    for i in range(self.num_conv_modules, self.num_total_modules):
      W, b = self.params["W%d" % (i+1)], self.params["b%d" % (i+1)]
      fc, fc_cache = affine_forward(relu, W, b)
      out_dict["fc%d_cache" % (i+1)] = fc_cache
      if i == self.num_conv_modules+self.num_hidden_layers-1:
        continue
      if self.use_batchnorm:
        gamma, beta = self.params["gamma%d" % (i+1)], self.params["beta%d" % (i+1)]
        bn, bn_cache = batchnorm_forward(fc, gamma, beta, 
                                         self.bn_params[self.num_conv_modules*self.num_conv_layers+(i-self.num_conv_modules)])
        out_dict["bn%d_cache" % (i+1)] = bn_cache
        relu, relu_cache = relu_forward(bn)
      else:
        relu, relu_cache = relu_forward(fc)
      if self.use_dropout:
        relu, dropout_cache = dropout_forward(relu, self.dropout_param)
        out_dict["dropout%d_cache" % (i+1)] = dropout_cache
      out_dict["relu%d_cache" % (i+1)] = relu_cache
    scores, scores_cache = fc, fc_cache
        
    if mode == "test":
      return scores
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores, y)
    for i in range(0, self.num_conv_modules):
      for j in range(0, self.num_conv_layers):
        W = self.params["W%d_%d" % (i+1, j+1)]
        loss += 0.5 * self.reg * np.sum(W * W)

    for i in range(self.num_conv_modules, self.num_total_modules):
      W = self.params["W%d" % (i+1)]
      loss += 0.5 * self.reg * np.sum(W * W)

    drelu, dW, db = affine_backward(dscores, scores_cache)
    for i in range(self.num_total_modules-1, self.num_conv_modules-1, -1):
      W, b = self.params["W%d" % (i+1)], self.params["b%d" % (i+1)]
      if not i == self.num_total_modules - 1:
        if self.use_dropout:
          dropout_cache = out_dict["dropout%d_cache" % (i+1)]
          drelu = dropout_backward(drelu, dropout_cache)
        relu_cache = out_dict["relu%d_cache" % (i+1)]
        dfc = relu_backward(drelu, relu_cache)
        if self.use_batchnorm:
          dbn = dfc
          bn_cache = out_dict["bn%d_cache" % (i+1)]
          dfc, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
          grads["gamma%d" % (i+1)], grads["beta%d" % (i+1)] = dgamma, dbeta
        fc_cache = out_dict["fc%d_cache" % (i+1)]
        drelu, dW, db = affine_backward(dfc, fc_cache)
      dW += self.reg * W
      grads["W%d" % (i+1)], grads["b%d" % (i+1)] = dW, db

    for i in range(self.num_conv_modules-1, -1, -1):
      dpool = drelu
      pool_cache = out_dict["pool%d_cache" % (i+1)]
      drelu = max_pool_backward_fast(dpool, pool_cache)
      for j in range(self.num_conv_layers-1, -1, -1):
        W, b = self.params["W%d_%d" % (i+1, j+1)], self.params["b%d_%d" % (i+1, j+1)]
        if self.use_dropout:
          dropout_cache = out_dict["dropout%d_%d_cache" % (i+1, j+1)]
          drelu = dropout_backward(drelu, dropout_cache)
        relu_cache = out_dict["relu%d_%d_cache" % (i+1, j+1)]
        dconv = relu_backward(drelu, relu_cache)
        if self.use_batchnorm:
          dbn = dconv
          bn_cache = out_dict["bn%d_%d_cache" % (i+1, j+1)]
          dconv, dgamma, dbeta = spatial_batchnorm_backward(dconv, bn_cache)
          grads["gamma%d_%d" % (i+1, j+1)], grads["beta%d_%d" % (i+1, j+1)] = dgamma, dbeta
        conv_cache = out_dict["conv%d_%d_cache" % (i+1, j+1)]
        drelu, dW, db = conv_backward_fast(dconv, conv_cache)
        dW += self.reg * W
        grads["W%d_%d" % (i+1, j+1)], grads["b%d_%d" % (i+1, j+1)] = dW, db
    
    return loss, grads
