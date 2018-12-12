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
  N,D=X.shape
  D,C=W.shape
  scores=X.dot(W)-np.max(X.dot(W),axis=1,keepdims=True)
  sums=np.exp(scores).sum(axis=1,keepdims=True)
  for n in range(N):
    for k in range(C):
      loss=loss-(y[n]==k)*(scores[n,k]-np.log(sums[n]))
      p=np.exp(scores)/sums
      dW[:,k]=dW[:,k]+(p[n,k]-(y[n]==k))*X[n,:]
  loss=loss+0.5*reg*np.sum(W*W)
  dW=dW+reg*W
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
  N,D=X.shape
  D,C=W.shape
  scores=X.dot(W)
  exp_scores=np.exp(scores-np.max(scores,axis=1,keepdims=True))
  probs=exp_scores/exp_scores.sum(axis=1,keepdims=True)
  loss=-np.log(probs[np.arange(N),y]).sum()+0.5*reg*np.sum(W*W)
  dprobs=probs
  dprobs[np.arange(N),y]-=1
  dW=X.T.dot(dprobs)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

