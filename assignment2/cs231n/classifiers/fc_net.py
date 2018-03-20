from builtins import range
from builtins import object
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

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        w1 = np.random.randn(input_dim,hidden_dim)*weight_scale
        b1 = np.zeros(hidden_dim,)
        w2 = np.random.randn(hidden_dim,num_classes)*weight_scale
        b2 = np.zeros(num_classes,)
        self.params['W1'] = w1
        self.params['W2'] = w2
        self.params['b1'] = b1
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        hid1,cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores,cache2 = affine_forward(hid1,self.params['W2'],self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss,dscore = softmax_loss(scores,y)
        loss+=0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
        dhid1,dw2,db2 = affine_backward(dscore, cache2)
        dx,dw1,db1 = affine_relu_backward(dhid1,cache1)
        grads['W1'] = dw1 + self.reg*self.params['W1']
        grads['W2'] = dw2 + self.reg*self.params['W2']
        grads['b1'] = db1
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
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
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        #######################################################################
        new_dim=[input_dim]+hidden_dims
        self.num_dim = len(new_dim)
        for i in range(len(new_dim)-1):
            self.params['W'+str(i+1)]=np.random.randn(new_dim[i],new_dim[i+1])*weight_scale
            self.params['b'+str(i+1)]=np.zeros(new_dim[i+1])
            if self.use_batchnorm == True:
                if i < len(hidden_dims):
                    self.params['beta'+str(i+1)] = np.zeros((1, new_dim[i+1]))
                    self.params['gamma'+str(i+1)] = np.ones((1, new_dim[i+1]))
        if self.use_dropout>0:
            self.dropout_param = {'p': dropout,'mode':'train','seed':seed}


        self.params['W'+str(len(new_dim))] = np.random.randn(new_dim[-1],num_classes)*weight_scale
        self.params['b'+str(len(new_dim))] = np.zeros(num_classes)
        self.new_dim = new_dim

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        #######################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        #######################################################################
        hids = [X]
        caches = []
        for i in range(len(self.new_dim)):
            if i != len(self.new_dim)-1:
                if self.use_batchnorm==True:
                    bn_param = {'mode':'train'}
                    hid,cache = affine_batch_relu_forward(hids[-1],self.params['W'+str(i+1)],self.params['b'+str(i+1)],self.params['gamma'+str(i+1)],self.params['beta'+str(i+1)],bn_param)
                elif self.use_dropout>0:
                    hid,cache = affine_relu_drop_forward(hids[-1],self.params['W'+str(i+1)],self.params['b'+str(i+1)],self.dropout_param)
                else:
                    hid,cache = affine_relu_forward(hids[-1],self.params['W'+str(i+1)],self.params['b'+str(i+1)])  	
            else:
                hid,cache = affine_forward(hids[-1],self.params['W'+str(i+1)],self.params['b'+str(i+1)])
                # print "sb",i,self.params['W'+str(i+1)].shape,hid.shape
            hids.append(hid)
            caches.append(cache)
        scores = hids[-1]
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.Good to know                                        #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #######################################################################
        loss,dscore = softmax_loss(hids[-1],y)
        for i in range(len(self.new_dim)):
            loss += 0.5*self.reg*np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
        grads={}
        dhid = dscore
        i = len(self.new_dim)
        while i>0:
            if i==len(self.new_dim):
                dhid,dw,db = affine_backward(dhid,caches[i-1])
            else:
                if self.use_batchnorm==True:
                    dhid,dw,db,dgamma, dbeta = affine_batch_relu_backward(dhid, caches[i-1])
                elif self.use_dropout>0:
                    dhid,dw,db = affine_relu_drop_backward(dhid, caches[i-1])
                else:
                    dhid,dw,db = affine_relu_backward(dhid,caches[i-1])            
            grads['W'+str(i)]=dw + self.reg*self.params['W'+str(i)]
            grads['b'+str(i)]=db
            if i != len(self.new_dim) and self.use_batchnorm==True:
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta
            i = i-1

        return loss, grads



def affine_batch_relu_forward(x, w, b,gamma,beta,bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bb, bt_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bb)
    cache = (fc_cache, bt_cache, relu_cache)
    return out, cache

def affine_batch_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bt_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    do, dgamma, dbeta = batchnorm_backward(da, bt_cache)
    dx, dw, db = affine_backward(do, fc_cache)
    return dx, dw, db,dgamma, dbeta


def affine_relu_drop_forward(x, w, b,dropout_param):
    a, fc_cache = affine_forward(x, w, b)
    r, relu_cache = relu_forward(a)
    out, dr_cache = dropout_forward(r, dropout_param)
    cache = (fc_cache, relu_cache,dr_cache)
    return out, cache

def affine_relu_drop_backward(dout, cache):
    fc_cache, relu_cache, dr_cache = cache
    do = dropout_backward(dout, dr_cache)
    da = relu_backward(do, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db