from builtins import object
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

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # set some hyperparameters
        stride = 1
        pad = (filter_size - 1) // 2
        pool_height = 2
        pool_width = 2
        pool_stride = 2
        C = input_dim[0]
        H = input_dim[1]
        W = input_dim[2]
        F = num_filters
        HH = WW = filter_size
        pad = (HH - 1) / 2
        H_out = 1 + (H + 2 * pad - HH) /stride
        W_out = 1 + (W + 2 * pad - WW) / stride
        H_out_pool = 1 + (H_out - pool_height) / pool_stride
        W_out_pool = 1 + (W_out - pool_width) / pool_stride
        x_axis = F*H_out_pool*W_out_pool
        self.params['W1'] = weight_scale*np.random.randn(F,C,HH,WW)
        self.params['b1'] = np.zeros(F)
        self.params['W2'] = weight_scale*np.random.randn(x_axis,hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
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
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out_conv,cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_af_rl,cache_af_rl = affine_relu_forward(out_conv, W2, b2)
        scores,cache_sc = affine_forward(out_af_rl, W3, b3) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        daf, dW3, db3 = affine_backward(dscores, cache_sc)
        dconv,dW2,db2 = affine_relu_backward(daf, cache_af_rl)
        dx,dW1,db1 = conv_relu_pool_backward(dconv, cache_conv)
        loss += 0.5*self.reg*np.sum(W1**2) + 0.5*self.reg*np.sum(W2**2) + 0.5*self.reg*np.sum(W3**2)
        grads['W1'] = dW1+self.reg*W1
        grads['b1'] = db1
        grads['W2'] = dW2+self.reg*W2
        grads['b2'] = db2
        grads['W3'] = dW3+self.reg*W3
        grads['b3'] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class MultiLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=[100,50], num_classes=10, weight_scale=1e-3, reg=0.0,
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

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # set some hyperparameters
        stride = 1
        pad = (filter_size - 1) // 2
        pool_height = 2
        pool_width = 2
        pool_stride = 2
        C = input_dim[0]
        H = input_dim[1]
        W = input_dim[2]
        F = num_filters
        HH = WW = filter_size
        pad = (HH - 1) / 2
        H_out = 1 + (H + 2 * pad - HH) /stride
        W_out = 1 + (W + 2 * pad - WW) / stride
        H_out_pool = 1 + (H_out - pool_height) / pool_stride
        W_out_pool = 1 + (W_out - pool_width) / pool_stride
        x_axis = F*H_out_pool*W_out_pool
        self.params['W1'] = weight_scale*np.random.randn(F,C,HH,WW)
        self.params['b1'] = np.zeros(F)
        self.params['W2'] = weight_scale*np.random.randn(x_axis,hidden_dim[0])
        self.params['b2'] = np.zeros(hidden_dim[0])
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim[0],hidden_dim[1])
        self.params['b3'] = np.zeros(hidden_dim[1])
        self.params['W4'] = weight_scale*np.random.randn(hidden_dim[1],num_classes)
        self.params['b4'] = np.zeros(num_classes)
        self.params['beta'] = np.zeros((1, H))
        self.params['gamma'] = np.ones((1, H))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma, beta = self.params['gamma'],self.params['beta']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        bn_param = {'mode':'train'}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out_conv,cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_batch,cache_batch = spatial_batchnorm_forward(out_conv, gamma, beta, bn_param)
        out_af_rl1,cache_af_rl1 = affine_relu_forward(out_batch, W2, b2)
        out_af_rl2,cache_af_rl2 = affine_relu_forward(out_af_rl1, W3, b3)
        scores,cache_sc = affine_forward(out_af_rl2, W4, b4) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        daf2, dW4, db4 = affine_backward(dscores, cache_sc)
        daf1, dW3, db3 = affine_relu_backward(daf2, cache_af_rl2)
        dconv, dW2, db2 = affine_relu_backward(daf1, cache_af_rl1)
        dbatch, dgamma,dbeta = spatial_batchnorm_backward(dconv, cache_batch)
        dx,dW1,db1 = conv_relu_pool_backward(dbatch, cache_conv)
        loss += 0.5*self.reg*np.sum(W1**2) + 0.5*self.reg*np.sum(W2**2) + 0.5*self.reg*np.sum(W3**2)+0.5*self.reg*np.sum(W4**2)
        grads['W1'] = dW1+self.reg*W1
        grads['b1'] = db1
        grads['W2'] = dW2+self.reg*W2
        grads['b2'] = db2
        grads['W3'] = dW3+self.reg*W3
        grads['b3'] = db3
        grads['W4'] = dW4+self.reg*W4
        grads['b4'] = db4
        grads['gamma'] = dgamma
        grads['beta'] = dbeta
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

