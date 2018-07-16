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
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params["b1"] = np.zeros((hidden_dim, ))
        self.params["W2"] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params["b2"] = np.zeros((num_classes, ))
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
        out1, cache1 = affine_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2 = relu_forward(out1)
        out3, cache3 = affine_forward(out2, self.params["W2"], self.params["b2"])
        scores = out3
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
        loss, dout = softmax_loss(scores, y)
        d3, grads["W2"], grads["b2"] = affine_backward(dout, cache3)
        d2 = relu_backward(d3, cache2)
        d1, grads["W1"], grads["b1"] = affine_backward(d2, cache1)

        loss += 0.5 * self.reg * (np.sum(self.params["W1"] * self.params["W1"])
                                  + np.sum(self.params["W2"] * self.params["W2"]))
        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
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
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.first = True

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        self.hidden_dims = hidden_dims[:]
        self.hidden_dims.insert(0, input_dim)
        self.hidden_dims.append(num_classes)
        for i in range(1, self.num_layers + 1):
            self.params[f"W{i}"] = np.random.normal(0, weight_scale,
                                                    (self.hidden_dims[i - 1], self.hidden_dims[i]))
            self.params[f"b{i}"] = np.zeros(self.hidden_dims[i])
            if (self.normalization == "batchnorm" or self.normalization == "layernorm") and i != self.num_layers:
                self.params[f"beta{i}"] = np.zeros(self.hidden_dims[i])
                self.params[f"gamma{i}"] = np.ones(self.hidden_dims[i])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
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
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
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
        ############################################################################

        num_loop = 3
        num_all_layers = self.num_layers * num_loop - 1
        outs = [None for i in range(num_all_layers)]
        caches = [None for i in range(num_all_layers)]

        outs[0] = X
        for i in range(1, num_all_layers, 1):
            if i % num_loop == 1 :
                j = i // num_loop + 1
                outs[i], caches[i] = affine_forward(outs[i - 1],
                                                    self.params[f"W{j}"],
                                                    self.params[f"b{j}"])
            elif i % num_loop == 2:
                j = i // num_loop
                if self.normalization == "batchnorm":
                   outs[i], caches[i] = \
                        batchnorm_relu_forward(outs[i - 1],
                                               self.params[f"gamma{j}"],
                                               self.params[f"beta{j}"],
                                               self.bn_params[j - 1])
                elif self.normalization == "layernorm":
                    outs[i], caches[i] =\
                        layernorm_relu_forward(outs[i - 1],
                                               self.params[f"gamma{j}"],
                                               self.params[f"beta{j}"],
                                               self.bn_params[j - 1])
                else:
                    outs[i], caches[i] = relu_forward(outs[i - 1])
            elif i % num_loop == 0:
                    if self.use_dropout:
                        outs[i], caches[i] = dropout_forward(outs[i - 1], self.dropout_param)
                    else:
                        outs[i], caches[i] = outs[i - 1], caches[i - 1]
        scores = outs[-1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        douts = [None for i in range(num_all_layers + 1)]
        loss, douts[-1] = softmax_loss(scores, y)
        douts[-1] = (douts[-1], )
        for i in range(num_all_layers - 1, 0, -1):
            if i % num_loop == 1:
                j = i // num_loop + 1
                douts[i] = affine_backward(douts[i + 1][0], caches[i])
                _, grads[f"W{j}"], grads[f"b{j}"] = douts[i]
                loss += 0.5 * self.reg * np.sum(self.params[f"W{j}"] * self.params[f"W{j}"])
                grads[f"W{j}"] += self.reg * self.params[f"W{j}"]
            elif i % num_loop == 2:
                j = i // num_loop
                if self.normalization == "batchnorm":
                    douts[i] = batchnorm_relu_backward(douts[i + 1][0], caches[i])
                    _, grads[f"gamma{j}"], grads[f"beta{j}"] = douts[i]
                elif self.normalization == "layernorm":
                    douts[i] = layernorm_relu_backward(douts[i + 1][0], caches[i])
                    _, grads[f"gamma{j}"], grads[f"beta{j}"] = douts[i]
                else:
                    douts[i] = (relu_backward(douts[i + 1][0], caches[i]), )
            elif i % num_loop == 0:
                if self.use_dropout:
                    douts[i] = (dropout_backward(douts[i + 1][0], caches[i]), )
                else:
                    douts[i] = douts[i + 1]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def batchnorm_relu_forward(x, gamma, beta, bn_params):
    """
    Convenience layer that perorms a batchnorm transform followed by a ReLU

    Inputs:
    - x: Input to the batchnorm layer
    - beta, gamma, bn_params: Parameters for the batchnorm layer.

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, bn_cache = batchnorm_forward(x, gamma, beta, bn_params)
    out, relu_cache = relu_forward(a)
    cache = (bn_cache, relu_cache)
    return out, cache


def batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the batchnorm-relu convenience layer
    """
    bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    return dx, dgamma, dbeta

def layernorm_relu_forward(x, gamma, beta, bn_params):
    """
    Convenience layer that perorms a layernorm transform followed by a ReLU

    Inputs:
    - x: Input to the batchnorm layer
    - beta, gamma, bn_params: Parameters for the batchnorm layer.

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, ln_cache = layernorm_forward(x, gamma, beta, bn_params)
    out, relu_cache = relu_forward(a)
    cache = (ln_cache, relu_cache)
    return out, cache

def layernorm_relu_backward(dout, cache):
    """
    Backward pass for the batchnorm-relu convenience layer
    """
    ln_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = layernorm_backward(da, ln_cache)
    return dx, dgamma, dbeta