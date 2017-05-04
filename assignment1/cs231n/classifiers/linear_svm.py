import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # count the number of classes that didnt' meet the desired margin 
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    indicator = (scores - correct_class_score + 1) > 0 
    for j in xrange(num_classes):
      if j == y[i]:
        # gradient taken from http://cs231n.github.io/optimization-1/
        dW[:,j] -= np.sum(np.append(indicator[:j],indicator[j+1:]))*X[i,:] 
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # true for j != yi
        dW[:,j] += indicator[j]*X[i,:]
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*2*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = X.shape[0]
  num_classes = W.shape[1]
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # obtain the right dimenions 
  correct_scores = np.repeat(scores[np.arange(num_train),y],num_classes).reshape(num_train,num_classes)
  matrix = scores - correct_scores + 1
  # set all true label positions to 0 
  matrix[np.arange(num_train),y] = 0
  # calculate the loss of each training example
  maximums = np.maximum(np.zeros((num_train,num_classes)),matrix)
  loss = np.sum(maximums)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  indicators = np.zeros(maximums.shape)
  indicators[maximums > 0] = 1    
  row_sum = np.sum(indicators, axis=1)
  indicators[range(num_train),y] = -row_sum
  dW = X.T.dot(indicators)

  # average and regularize both the loss/gradient 
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train 
  dW += reg*2*W 

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
