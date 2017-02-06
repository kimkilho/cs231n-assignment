import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # Shift the values of scores so that the highest number is 0
    scores -= np.max(scores)  
    softmax_scores_sum = np.sum(np.exp(scores))
    softmax_scores = np.exp(scores)/softmax_scores_sum

    for j in xrange(num_classes):
      if j == y[i]:
        loss += -np.log(softmax_scores[j])
        dW[:, j] += X[i].T * -(1-softmax_scores[j])
      else:
        dW[:, j] += X[i].T * softmax_scores[j]

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # Convert y to follow one-hot encoding
  y_one_hot = np.zeros((len(y), W.shape[1]))
  y_one_hot[np.arange(len(y)), y] = True

  scores = X.dot(W)   # (N, C)
  scores -= np.expand_dims(np.max(scores, axis=1), axis=1)
  softmax_scores_sum = np.expand_dims(np.sum(np.exp(scores), axis=1), axis=1) # (N, 1)
  softmax_scores = np.exp(scores) / softmax_scores_sum  # (N, C)

  correct_class_softmax_scores = \
    softmax_scores[np.arange(len(softmax_scores)), y]   # (N, 1)
  loss = np.sum(-np.log(correct_class_softmax_scores)) / X.shape[0]

  softmax_scores[np.arange(X.shape[0]), y] -= 1
  dW += X.T.dot(softmax_scores) # (D, C)
    
  dW /= X.shape[0]

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW

