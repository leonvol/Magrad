<p align="center">
  <img src="https://raw.githubusercontent.com/leonvol/magrad/master/docs/magrad_large.png">
</p>


[![Build Status](https://travis-ci.com/leonvol/Magrad.svg?branch=master)](https://travis-ci.com/leonvol/Magrad)
[![Coverage](https://codecov.io/gh/leonvol/Magrad/branch/master/graph/badge.svg)](https://codecov.io/gh/leonvol/Magrad)


This is magrad, a tensor autograd/machine learning framework written completely in Julia.

Although the magrad project follows the typical Julia package structure, it is not named Magrad.jl as it should not be confused as an officially registered package.

# status of the project
This project is a work in progress.

These are some goals for the future:
- implement gradient tape to keep track of computation
- fully support backward pass on broadcasted functions
- implement optimizer (?)

# workings of magrad & documentation
Magrad is a very simple and small project. The general workings can be described in a couple of sentences.
The core part of the project is the `Tensor` class (or struct in Julia). The `Tensor` holds all the information, gradients and currently also keeps track of the computation of itself, although I will split functionality and create a gradient tape for that purpose in the future. The actual computation of the gradients handled by the `backward()` function which traverses the computation graph saved in the Tensors in reverse order and applies the chain rule repeatedly. And that's all there is to it. 

Every critical piece of code has a comment attached, describing its function and design decisions. This is the general structure of the project: 


| file name | function |
| ----------| ---------|
| `Magrad.jl` | magrad library, includes `tensor.jl` |
| `tensor.jl` | implementation of the Tensor type, includes all of the following files to expose all the needed functionality to the user |
| `ops.jl` | implementation of the currently supported operations like +, -, matmul of the Tensor type|
| `grad.jl` | dispatched methods to calculate gradients |
| `broadcast.jl` | methods to support broadcasting for the `Tensor` type, not finished yet, broadcasting is not fully supported for the gradient calculation step |

# usage
magrad is easy to use and can prepare you for working with bigger frameworks like PyTorch. Here a small showcase of the basic functionality: 
```
using Magrad: Tensor, backward

a = Tensor(ones(3,3))
res = 3 .* a # supports +,-,*,.+,.-,.*
backward(res) # a.grad = [3 3 3; 3 3 3; 3 3 3]
```

# testing
There are tests for every critical part of the code, tested with specific examples against PyTorch.

To run tests locally, make sure to have activated the magrad environment and run the following:
```
include("test/runtests.jl")
```