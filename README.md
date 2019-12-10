# Option Critic
This repository is a PyTorch implementation of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup. It is mostly a rewriting of the original Theano code found (https://github.com/jeanharb/option_critic)[here] into PyTorch.

It is currently a work in progress, but it works for feature based environments (such as CartPole).

## Feature based deep-option critic
Currently, the feature based model can learn CartPole-v0 with a learning rate of 0.005, this has however only been tested with two options. (I dont see any reason to use more than two in the cart pole environment.) the current runs directory holds the training results for this env with 0.005 and 0.006 learning rates.

## Todo's

- [ ] The convolutional approach should be very similar, but will require some parameter search
- [ ] A simple baseline might be the four rooms experiment, but rendered instead of tabular.

## Requirements

```
pytorch 1.3.0
tensorboard 2.0.2
gym 0.15.3
```

## Changes with respect to the original implementation
- Using only one optimizer. It doesn't 

## some extra stuff i noticed in the theano implementation
- the authors experimented with double Q learning
- the Q network has a fixed/frozen target network used for more stable Q approximations (less oscillations)