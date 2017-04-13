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
	dW_ = np.zeros((500,10))

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in xrange(num_train):
		scores = X[i].dot(W)
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, keepdims=True)

		dscores = probs
		for j in xrange(num_classes):
			if j == y[i]:
				corect_logprobs = -np.log(probs[y[i]])
				loss += corect_logprobs
				dscores[y[i]] -= 1
		dW_[i] = dscores

	loss /= num_train
	dW_ /= num_train


	loss += 0.5 * reg * np.sum(W * W)
	
	dW = np.dot(X.T, dW_)
	dW += reg * W
	
	
	
	#pass
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
	scores = np.dot(X, W)
	exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	#corect_logprobs = -np.log(probs[range(X.shape[0]),y])
	#data_loss = np.sum(corect_logprobs)/X.shape[0]
	#reg_loss = 0.5 * 0 * np.sum(W*W)
	#loss = data_loss + reg_loss
	loss = -np.log(probs[np.arange(y.shape[0]), y]).mean()
	
	dscores = probs
	dscores[range(X.shape[0]),y] -= 1

	dW = np.dot(X.T, dscores)
	dW /= X.shape[0]
	dW += reg * W
	
	pass
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW

