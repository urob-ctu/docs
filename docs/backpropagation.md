---
title: Backpropagation
layout: default
has_children: false
nav_order: 3
mathjax: true
---

# Backpropagation
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Introduction

Backpropagation is a method used in artificial neural networks to calculate the gradient of the loss function with respect to the weights of the network. It is a supervised learning algorithm, which means that it requires labeled data to train the network. The backpropagation algorithm is the foundation of training neural networks and is used in many different neural network architectures.

This may seem like a complex concept, but we will break it down into simpler parts to understand it better.

## The Computation graph

The computation graph is a visual representation of the computations that are performed in a neural network. It is a directed acyclic graph that shows how the input data is transformed as it passes through the network.

Depending on the architecture of the network, the computation graph can be simple or complex. For example, a simple neuron with $n$ inputs and one output can be represented in the following way:

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/neuron_graph.png" width="800">
</div>

We can easily visualize the forward and backward pass of the network using the computation graph.

## The Forward Pass

The forward pass is the first step in the backpropagation algorithm. In the forward pass, the input data is passed through the network, and the output is calculated. The output is then compared to the true output, and the error is calculated using a loss function. The loss function is a measure of how far the predicted output is from the true output.

In the computational graph it is presented as calculation of the output of the network. It is represented by arrows that go from left to right.

## The Backward Pass

The backward pass is the second step in the backpropagation algorithm. In the backward pass, the gradient of the loss function with respect to the weights of the network is calculated.

The backward pass the chain-rule is used to calculate the gradients as we propagate through the network.

## Updating the Weights

After the gradients are calculated, the weights of the network are updated using an optimization algorithm. The most common optimization algorithm used in neural networks is stochastic gradient descent (SGD). The weights are updated in the opposite direction of the gradient to minimize the loss function.

Many variations of the SGD algorithm exist, such as Adam, RMSprop, and Adagrad, which have different ways of updating the weights. More about these algorithms can be found [here](https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0)

## Vector-Jacobian Product

TODO

## Expected Knowledge

From this text, you should understand the following concepts:

- **Computational graph**: A visual representation of the computations that are performed in a neural network.
- **Forward pass**: The step in the backpropagation algorithm where the input data is passed through the network, and the output is calculated.
- **Backward pass**: The step in the backpropagation algorithm where the gradient of the loss function with respect to the weights of the network is calculated.
- **Vector-Jacobian product**: An efficient way to calculate the gradients in the backward pass using the chain rule.
