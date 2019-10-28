# PyTorch Implementation of Jacobian Regularization

This library provides a PyTorch implementation of the Jacobian Regularization described in the paper "Robust Learning with Jacobian Regularization"
[arxiv:1908.02729](https://arxiv.org/abs/1908.02729). 

Jacobian regularization is a model-agnostic way of increasing classification margins, improving robustness to white and adversarial noise without severely hurting clean model performance. The implementation here also automatically supports GPU acceleration. 

For additional information, please see [1].



<p align="center">
    <img src="./figures/margins.png" alt="Classification margins for different regularizers" height="250" />
</p>

---

## Installation
```
pip install git+https://github.com/facebookresearch/jacobian_regularizer
```

## Usage
This library provides a simple subclass of `torch.nn.Module` that implements Jacobian regularization. After installation, first import the regularization loss
```python
from jacobian import JacobianReg
import torch.nn as nn
```
where we have also imported `torch.nn` so that we may also include a standard supervised classification loss.

To use Jacobian regularization, we initialize the Jacboan regularization at the same time we initialize our loss criterion
```python
criterion = nn.CrossEntropyLoss() # supervised classification loss
reg = JacobianReg() # Jacobian regularization
lambda_JR = 0.01 # hyperparameter
```
where we have also included a hyperparameter `lambda_JR` controlling the relative strength of the regularization.

Let's assume we also have a model `model`, data loader `loader`, optimizer `optimizer` and `device` is either `torch.device("cpu")` for CPU training or `torch.device("cuda:0")` for GPU training. Then, to use Jacobian regularization, our training loop might look like this
```python
for idx, (data, target) in enumerate(loader):

    data, target = data.to(device), target.to(device)
    data.requires_grad = True # this is essential!

    optimizer.zero_grad()

    output = model(data) # forward pass

    loss_super = criterion(output, target) # supervised loss
    R = reg(data, output)   # Jacobian regularization
    loss = loss_super + lambda_JR*R # full loss

    loss.backward() # computes gradients

    optimizer.step()
```
Backpropagation of the full loss occurs in the call `loss.backward()` so long as  `data.requires_grad = True` was called at the top of the training loop. **Note:** this is important any time the Jacobian regularization is evaluated, whether doing model training or model evaluation. (Even for just computing the Jacobian loss, gradients are required!)

As implied, this Jacobian regularization is compatible with both CPU and GPU training, and may also be combined with other losses, regularizations, and will work with any model, optimizer, or dataset.

### Keyword Arguments
 - n (int, optional): determines the number of random projections. If n=-1, then it is set to the dimension of the output space and projection is non-random and orthonormal, yielding the exact result.  For any reasonable batch size, the default (n=1) should be sufficient.
```python
  reg = JacobianReg() # default has 1 projection

  # you can also specify the number of projections
  # this should be must less than the number of classes
  n_proj = 3
  reg_proj = JacobianReg(n=n_proj)

  # alternatively, you can get the full Jacobian
  # which takes C times as long as n_proj=1, if C is # of classes
  reg_full = JacobianReg(n=-1) 
```

## Examples
An example script that uses Jacobian regularization for simple MLP training on MNIST is given in the  [`examples`](./examples) directory in the file [`mnist.py`](./examples/mnist.py). If you execute the script after installing this package
```python
python mnist.py
```
you should start to see output like this
```
Training epoch 1.
[1,   100] supervised loss: 0.687, Jacobian loss: 3.383
[1,   200] supervised loss: 0.373, Jacobian loss: 2.128
[1,   300] supervised loss: 0.317, Jacobian loss: 1.769
[1,   400] supervised loss: 0.287, Jacobian loss: 1.553
[1,   500] supervised loss: 0.276, Jacobian loss: 1.459
```
showing the Jacobian beginning to decrease as well as the supervised loss. After 5 epochs, the training will conclude and the output will show an evaluation on the test set before and after training
```
Test set results on MNIST with lambda_JR=0.100.

Before training:
  accuracy: 827/10000=0.083
  supervised loss: 2.675
  Jacobian loss: 3.656
  total loss: 3.041
  
After 5 epochs of training:
  accuracy: 9702/10000=0.970
  supervised loss: 0.027
  Jacobian loss: 0.977
  total loss: 0.125
```
showing that the model will learn to generalize and at the same time will regularize the Jacobian for greater robustness.

Please look at the example file [`mnist.py`](./examples/mnist.py) for additional details.

## License
jacobian_regularizer is licensed under the MIT license found in the LICENSE file.

## References
[1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida, "Robust Learning with Jacobian Regularization," 2019. [arxiv:1908.02729 [stat.ML]](https://arxiv.org/abs/1908.02729)

---

If you found this useful, please consider citing
```
@article{hry2019jacobian,
      author         = "Hoffman, Judy and Roberts, Daniel A. and Yaida, Sho",
      title          = "Robust Learning with Jacobian Regularization",
      year           = "2019",
      eprint         = "1908.02729",
      archivePrefix  = "arXiv",
      primaryClass   = "stat.ML",
}
```
