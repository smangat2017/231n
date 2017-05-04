import numpy as np
from random import shuffle
from past.builtins import xrange

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
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    f_i = X[i].dot(W)
    # using trick from notes 
    f_i -= np.max(f_i)
    sum_j = np.sum(np.exp(f_i))
    # declare a lambda function to calculate the loss for each point 
    p = lambda z: np.exp(f_i[z])/sum_j
    loss -= np.log(p(y[i]))
    # calculate the gradient 
    for c in range(num_classes):
        dW[:,c] += (p(c) - (c == y[i])) * X[i]
  
  # average out the loss and appropriately regularize 
  loss /= num_train
  dW /= num_train
  loss += .5*reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # follow same formulas used in previous part
  fs = X.dot(W)
  # keep dims enables proper broadcasting
  fs -= np.max(fs, axis=1, keepdims=True)
  sum_js = np.sum(np.exp(fs), axis=1, keepdims=True)
  p = np.exp(fs)/sum_js
    
  logs = -np.log(p[np.arange(num_train), y])
  loss = np.sum(logs)
  
  true_vals = np.zeros(p.shape)
  true_vals[np.arange(num_train),y] = 1
  dW = X.T.dot(p - true_vals)

  # average + regularize the loss/gradient
  loss /= num_train
  dW /= num_train
  loss += .5*reg*np.sum(W*W)
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

