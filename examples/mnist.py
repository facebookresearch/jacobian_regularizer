'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    Example script training a simple MLP on MNIST
    demonstrating the PyTorch implementation of
    Jacobian regularization described in [1].

    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
'''
from __future__ import division
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from jacobian import JacobianReg

class MLP(nn.Module):
    '''
    Simple MLP to demonstrate Jacobian regularization.
    '''
    def __init__(self, in_channel=1, im_size=28, num_classes=10, 
                 fc_channel1=200, fc_channel2=200):
        super(MLP, self).__init__()
        
        # Parameter setup
        compression=in_channel*im_size*im_size
        self.compression=compression
        
        # Structure
        self.fc1 = nn.Linear(compression, fc_channel1)
        self.fc2 = nn.Linear(fc_channel1, fc_channel2)
        self.fc3 = nn.Linear(fc_channel2, num_classes)
        
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = x.view(-1, self.compression)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def eval(device, model, loader, criterion, lambda_JR):
    '''
    Evaluate a model on a dataset for Jacobian regularization

    Arguments:
        device (torch.device): specifies cpu or gpu training
        model (nn.Module): the neural network to evaluate
        loader (DataLoader): a loader for the dataset to eval
        criterion (nn.Module): the supervised loss function
        lambda_JR (float): the Jacobian regularization weight

    Returns:
        correct (int): the number correct
        total (int): the total number of examples
        loss_super (float): the supervised loss
        loss_JR (float): the Jacobian regularization loss
        loss (float): the total combined loss
    '''

    correct = 0
    total = 0 
    loss_super_avg = 0 
    loss_JR_avg = 0 
    loss_avg = 0

    # for eval, let's compute the jacobian exactly
    # so n, the number of projections, is set to -1.
    reg_full = JacobianReg(n=-1) 
    for data, targets in loader:
        data = data.to(device)
        data.requires_grad = True # this is essential!
        targets = targets.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        loss_super = criterion(output, targets) # supervised loss
        loss_JR = reg_full(data, output) # Jacobian regularization
        loss = loss_super + lambda_JR*loss_JR # full loss
        loss_super_avg += loss_super.item()*targets.size(0)
        loss_JR_avg += loss_JR.item()*targets.size(0)
        loss_avg += loss.item()*targets.size(0)
    loss_super_avg /= total
    loss_JR_avg /= total
    loss_avg /= total
    return correct, total, loss_super.item(), loss_JR.item(), loss.item()

def main():
    '''
    Train MNIST with Jacobian regularization.
    '''
    seed = 1
    batch_size = 64
    epochs = 5

    lambda_JR = .1

    # number of projections, default is n_proj=1
    # should be greater than 0 and less than sqrt(# of classes)
    # can also set n_proj=-1 to compute the full jacobian
    # which is computationally inefficient
    n_proj = 1 

    # setup devices
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    # load MNIST trainset and testset
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std)]
    )
    trainset = datasets.MNIST(root='./data', train=True, 
        download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testset = datasets.MNIST(root='./data', train=False, 
        download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True
    )

    # initialize the model
    model = MLP()
    model.to(device)

    # initialize the loss and regularization
    criterion = nn.CrossEntropyLoss()
    reg = JacobianReg(n=n_proj) # if n_proj = 1, the argument is unnecessary

    # initialize the optimizer
    # including additional regularization, L^2 weight decay
    optimizer = optim.SGD(model.parameters(), 
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    # eval on testset before any training
    correct_i, total, loss_super_i, loss_JR_i, loss_i = eval(
        device, model, testloader, criterion, lambda_JR
    )

    # train
    for epoch in range(epochs):
        print('Training epoch %d.' % (epoch + 1) )
        running_loss_super = 0.0
        running_loss_JR = 0.0
        for idx, (data, target) in enumerate(trainloader):        

            data, target = data.to(device), target.to(device)
            data.requires_grad = True # this is essential!

            optimizer.zero_grad()

            output = model(data) # forward pass

            loss_super = criterion(output, target) # supervised loss
            loss_JR = reg(data, output)   # Jacobian regularization
            loss = loss_super + lambda_JR*loss_JR # full loss

            loss.backward() # computes gradients

            optimizer.step()

            # print running statistics
            running_loss_super += loss_super.item()
            running_loss_JR += loss_JR.item()
            if idx % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] supervised loss: %.3f, Jacobian loss: %.3f' %
                        (
                            epoch + 1, 
                            idx + 1, 
                            running_loss_super / 100,  
                            running_loss_JR / 100, 
                        )
                )
                running_loss_super = 0.0
                running_loss_JR = 0.0

    # eval on testset after training
    correct_f, total, loss_super_f, loss_JR_f, loss_f = eval(
        device, model, testloader, criterion, lambda_JR
    )

    # print results
    print('\nTest set results on MNIST with lambda_JR=%.3f.\n' % lambda_JR)
    print('Before training:')
    print('\taccuracy: %d/%d=%.3f' % (correct_i, total, correct_i/total))
    print('\tsupervised loss: %.3f' % loss_super_i)
    print('\tJacobian loss: %.3f' % loss_JR_i)
    print('\ttotal loss: %.3f' % loss_i)

    print('\nAfter %d epochs of training:' % epochs)
    print('\taccuracy: %d/%d=%.3f' % (correct_f, total, correct_f/total))
    print('\tsupervised loss: %.3f' % loss_super_f)
    print('\tJacobian loss: %.3f' % loss_JR_f)
    print('\ttotal loss: %.3f' % loss_f)

if __name__ == '__main__':
    main()
