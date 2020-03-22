# Option Critic
This repository is a PyTorch implementation of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup [arXiv](https://arxiv.org/abs/1609.05140). It is mostly a rewriting of the original Theano code found [here](https://github.com/jeanharb/option_critic) into PyTorch.


## Feature based deep-option critic
Currently, the feature based model can learn CartPole-v0 with a learning rate of 0.005, this has however only been tested with two options. (I dont see any reason to use more than two in the cart pole environment.) the current runs directory holds the training results for this env with 0.005 and 0.006 learning rates.

I suspect it will *only* take a grid search over learning rate to work on Pong and such. Just supply the right
```--env```
argument and the model should switch between features and convolutions.

## Four Room experiment
There are plenty of resources to find a numpy version of the four rooms experiment, this one is a little bit different; represent the state as a one-hot encoded vector, and learn to solve this grid world using a deep net. To enable this experiment, toggle
```python main.py --switch-goal True --env fourrooms```

## Requirements

```
pytorch 1.3.0
tensorboard 2.0.2
gym 0.15.3
```

## Changes with respect to the original implementation
- Using only one optimizer (RMSProp) for both acto and critic.
