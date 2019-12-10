# Option Critic
This repository is a PyTorch implementation of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup.

It is currently still a work in progress.

## Feature based deep-option critic
Currently, the feature based model can learn CartPole-v0 with a learning rate of 0.005, this has however only been tested with two options. (I dont see any reason to use more than two in the cart pole environment.) the current runs directory holds the training results for this env with 0.005 and 0.006 learning rates.


## some extra stuff i noticed in the theano implementation
- the authors experimented with double Q learning
- the Q network has a fixed/frozen target network used for more stable Q approximations (less oscillations)