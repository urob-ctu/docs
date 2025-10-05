---
title: Backpropagation
layout: default
nav_order: 3
parent: Training
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

Backpropagation is a cornerstone algorithm in the field of machine learning, particularly in the training of neural networks. It serves as the engine that enables the training of deep learning models. 

Conceptually, backpropagation is an algorithm for assigning "blame" or "responsibility" for the final loss to every parameter in the network. It starts with the total error at the end and works backward, using the chain rule to calculate how much each weight contributed to that error.

In this text, we will explain the backpropagation algorithm and its components.

{: .note }
>Note that we will not explain every term used in this text. We assume that you are familiar with the basic concepts of neural networks. If you are not yet, we recommend reading the previous texts in this course.

## The Computation graph

The computation graph is a visual representation of the computations that are performed in a neural network. It is a directed acyclic graph that shows how the input data is transformed as it passes through the network.

Depending on the architecture of the network, the computation graph can be simple or complex. For example, a simple neuron with $n$ inputs and one output can be represented in the following way:

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/neuron_graph.png" width="800">
</div>

We can easily visualize the forward and backward pass of the network using the computation graph.

## Components of the Backpropagation Algorithm

The backpropagation algorithm consists of three main components: the forward pass, the backward pass, and updating the weights. The forward pass is sometimes not considered a part of the backpropagation algorithm, but it is essential for understanding how the algorithm works.

We will explain each of these components in more detail and we will use this simple neural network as an example.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network.png" width="800">
</div>

$$ \boldsymbol{x}=\begin{bmatrix}{0.5 \\ 0.7}\end{bmatrix} \bar{\boldsymbol{w}}=\begin{bmatrix}{1.0 \\ 0.7 \\ 2.0}\end{bmatrix} z^{*}=3.5 $$

### The Forward Pass

The forward pass is the first step in the backpropagation algorithm. In the forward pass, the input data is passed through the network, and the output is calculated. The output is then compared to the true output, and the error is calculated using a loss function. The loss function is a measure of how far the predicted output is from the true output.

In the computational graph it is presented as calculation of the output of the network. It is represented by arrows that go from left to right.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network_fwd.png" width="800">
</div>

### The Backward Pass

The backward pass is the second step in the backpropagation algorithm. In the backward pass, the gradient of the loss function with respect to the weights of the network is calculated.

In the backward pass the chain-rule is used to calculate the gradients as we propagate through the network.

{: .definition }
>**Chain Rule** is a formula that expresses the derivative of the composition of the differentiable functions $f$ and $g$ in terms of the derivatives of $f$ and $g$. More precisely, if $h = f \circ g$ is the function such that $h(x) = f(g(x))$ for every $x$, then the chain rule is, in Lagrange's notation:
>
>$$h'(x) = f'(g(x)) g'(x)$$.
>
>In Leibniz's notation (if $y = f(z)$ and $z = g(x)$):
>
>$$\frac{dy}{dx} = \frac{dy}{dz} \frac{dz}{dx}$$.

In the computational graph it is presented as calculation of the gradients of the loss function with respect to the weights of the network. It is represented by arrows that go from right to left.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network_bwd.png" width="800">
</div>

### Updating the Weights

After the gradients are calculated, the weights of the network are updated using an optimization algorithm. The most common optimization algorithm used in neural networks is stochastic gradient descent (SGD). The weights are updated in the opposite direction of the gradient to minimize the loss function.

Many variations of optimization algorithms exist, such as Adam, RMSprop, and Adagrad, which have different ways of updating the weights. More about these algorithms can be found [here](https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0)

In the computational graph it is presented as updating the weights of the network. We will use the standard gradient descent with $\alpha=1.0$.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network_update.png" width="800">
</div>

## Vector-Jacobian Product

The vector-jacobian product is a way to efficiently calculate the gradient for large matrices. It utilizes linear algebra and matrix multiplication.

It is more efficient to calculate the gradient of the loss function with respect to the weights of the network using the vector-jacobian product as the jacobian matrix can be very large for high-dimensional data. The vector-jacobian product allows us to calculate the gradient without explicitly calculating the jacobian matrix. It also preserves the dimensionality of the input data in the backward pass.

{: .definition }
>The **vector-jacobian** product is defined as the following snippet (note that we will mix pseudocode with mathematical expresions)
>
>def $\textrm{vjp}_{f}(\boldsymbol{v}, \boldsymbol{z}):$
>
>&nbsp;&nbsp;&nbsp;&nbsp;return $\boldsymbol{v}^{\top}\cdot\frac{\partial f}{\partial\boldsymbol{z}}$

The primary advantage of the VJP approach is its efficiency. For a function mapping $$\mathbb{R}^n \rightarrow \mathbb{R}^m$$, the full Jacobian matrix has $$m \times n$$ elements. For a deep neural network, this is an enormous matrix that would be impossible to store in memory. The VJP calculates the required gradient contribution without ever forming this full matrix.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/vjp_graph.png" width="800">
</div>

## Expected Knowledge

Answer the following questions to test your understanding of backpropagation.

1. **Conceptual Understanding:** Explain backpropagation in non-mathematical terms. What is a "computation graph," and how does backpropagation use it to compute gradients?

2. **The Chain Rule:** Imagine a simple computation: $$z = w \cdot x + b$$, followed by $$a = \text{ReLU}(z)$$, and a loss $$L = (a_{true} - a)^2$$. Using the chain rule, write down the expression for the partial derivative of the loss with respect to the weight, $$\frac{\partial L}{\partial w}$$.

3. **Vector-Jacobian Product (VJP):** What is the main practical advantage of using the Vector-Jacobian Product (VJP) to implement backpropagation in deep learning libraries like PyTorch or TensorFlow, compared to explicitly calculating the entire Jacobian matrix?

4. **Forward vs. Backward Pass:** What key quantity is computed during the **forward pass**? What key quantity is computed for each parameter during the **backward pass**? How do these two passes work together in the context of gradient descent?

In practice, the backpropagation algorithm is implemented using automatic differentiation libraries such as TensorFlow and PyTorch. These libraries handle the computation of the gradients automatically, so you don't have to worry about the details of the backpropagation algorithm, while using them. However, understanding the backpropagation algorithm is essential for understanding how neural networks are trained.
