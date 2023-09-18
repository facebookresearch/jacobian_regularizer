'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    PyTorch implementation of Jacobian regularization described in [1].

    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
'''
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np


class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()
    '''
    The shape of the output tensor y is expected to be (B,...) where B is the batch_size.
    '''
    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2 if n=-1 or it approximates it
        '''
        shape = y.shape
        dimension = np.prod(shape[1:])
        if self.n == -1 or self.n>=dimension:
            # orthonormal tensor, sequentially spanned, of shape (dimension, shape)
            v = torch.eye(dimension)
            v = torch.reshape(v, (dimension,)+shape[1:])
            v = torch.unsqueeze(v,dim=1)
            v = v.expand((dimension,)+shape)

        else:
            # random properly-normalized vector for each sample
            v = self._random_vector(shape, dimension)
        if x.is_cuda:
            v = v.cuda()
        #v always has shape (num_proj, shape)
        Jv = grad(y, x, v, retain_graph=True, create_graph=True, is_grads_batched=True)[0]
        Jv = torch.square(Jv)
        Jv = torch.sum(Jv, dim = tuple(range(2,len(shape))))
        return dimension*torch.mean(Jv)/2

    def _random_vector(self, shape, dimension):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        v = torch.randn((self.n, shape[0]*dimension))
        v = torch.nn.functional.normalize(v, dim=-1)
        return v.reshape((self.n,)+shape)
