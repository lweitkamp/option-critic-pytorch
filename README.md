# Option Critic
This repository is a PyTorch implementation of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup.

It is currently still a work in progress.


## some extra stuff i noticed in the theano implementation
- the authors experimented with double Q learning
- the Q network has a fixed target network used for more stable Q approximations (less oscillations)