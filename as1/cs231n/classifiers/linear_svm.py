import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count_margin_over_zero = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count_margin_over_zero += 1
        dW[:, j] += X[i]

    dW[:, y[i]] -= count_margin_over_zero*X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

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
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # Convert y to follow one-hot and zero-hot encoding
  y_one_hot = np.zeros((len(y), W.shape[1]))
  y_one_hot[np.arange(len(y)), y] = True

  scores = X.dot(W)
  correct_class_scores = np.reshape(scores[np.arange(len(scores)), y], (len(scores), -1))

  margin = scores - correct_class_scores + 1
  incorrect_class_margin = margin * (y_one_hot == False)
  count_margin_over_zero = np.reshape(np.sum(incorrect_class_margin > 0, axis=1), (len(margin), -1))

  # dW increment for incorrect classes for each instances where margin > 0
  dW += X.T.dot((margin > 0) * (y_one_hot == False))

  # dW decrement for correct classes for each instances
  dW -= X.T.dot(count_margin_over_zero * y_one_hot)

  # HACK: Subtract 1*X.shape[0] to consider the case when margin at column j = y[i] is 1
  loss = (np.sum((margin > 0)*margin) - 1*X.shape[0]) / X.shape[0]

  loss += 0.5 * reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W

  return loss, dW
