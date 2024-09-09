---
title: Training Classifiers
layout: default
nav_order: 4
parent: Classification
---

# Training Classifiers
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

Throughout our exploration of classifiers, we've frequently referenced the concept of training. While there are straightforward methods like storing the entire dataset for algorithms like k-NN, these methods are often inefficient. Effective training of classifiers is a complex process that can be extremely costly. For instance, training some of the largest and most well-known neural networks, such as GPT models, can cost millions of U.S. dollars. Understanding the training process is crucial, and the most commonly used training algorithms are based on **gradient descent**. Gradient descent is an iterative technique used to minimize a **loss function**.

## Loss Function

To enhance our classifier, we need a way to measure its performance. This measurement is provided by a loss function. The loss function takes the **logits** (the raw prediction scores from the classifier) and the **true label** of the data as inputs and outputs a single number indicating how well the classifier is performing. The loss function is essential because it guides us in adjusting the classifier's parameters to improve its performance.

{: .definition }
> **Loss Function** A function that evaluates the performance of a classifier. It takes the logits and the true labels as input and returns a single number representing the classifier's performance. The goal is to minimize this loss function to enhance the classifier's accuracy.
>
> $$\mathcal{L}(\boldsymbol{s}, \boldsymbol{y}) = \ell, \quad \boldsymbol{s} \in \mathbb{R}^{c}, \quad \boldsymbol{y} \in \mathbb{R}^{c}$$
>
> Here the $$\boldsymbol{s}$$ represents the logits of the classifier, and the $$y$$ represents the true label in one-hot vector form. The loss function outputs a scalar value $$\ell$$ that quantifies how well the classifier is performing.

It's important to note that the loss function typically requires the true label in a one-hot encoded format, which will be explained further in the section on [loss]({{ site.baseurl }}{% link docs/classification/loss.md %}).

## Gradient Descent

Gradient descent is a fundamental optimization technique used in training classifiers. It works by iteratively adjusting the parameters (weights and biases) of the classifier to minimize the loss function. The key idea is to compute the gradient of the loss function with respect to the parameters and update the parameters in the opposite direction of the gradient. This iterative process continues until convergence, where the loss function is minimized or a stopping criterion is met.

<br>
<div align="center">
<figure>
      <img src="{{ site.baseurl }}/assets/images/gradient_descent.gif" width="250px"/>
      <figcaption><strong>Gradient Descent Visualization</strong>: Taken from <a href="https://commons.wikimedia.org/wiki/File:Gradient_descent.gif">Wikimedia</a></figcaption>
</figure>
</div>
<br>

### Key Steps in Gradient Descent:

1. **Compute the Gradient**: Calculate the gradient of the loss function with respect to each parameter (weight and bias). The gradient indicates the direction and magnitude of the steepest increase in the loss.

2. **Update the Parameters**: Adjust each parameter in the direction that reduces the loss function. This adjustment is proportional to the gradient and a predefined learning rate, which controls the size of each update step.

3. **Iterate**: Repeat the process of computing gradients and updating parameters until the loss function converges to a minimum or a stopping condition is satisfied (e.g., maximum number of iterations reached).

Gradient descent is efficient for optimizing complex models with large amounts of data, though it requires careful tuning of the learning rate and monitoring for convergence to avoid issues like slow convergence or overshooting the optimal solution.

## Forward Pass and Backward Pass

The [prediction function]({{ site.baseurl }}{% link docs/classification/logits.md %}#predictor_detailed), which we previously discussed in the context of generating predictions, is part of the broader training process. The training function $$\mathcal{T}(\boldsymbol{x}, y)$$ extends this concept by incorporating the true labels into the process. Training involves two main computations: the **Forward Pass** and the **Backward Pass**.

### Forward Pass

In the forward pass the loss is computed given the input $$\boldsymbol{x}x$$ and the true label $$y$$. This is computed using the weights $$\boldsymbol{w}$$ as discussed before. Output of the forward pass is the loss $$\ell$$. 

<br>
<div align="center">
      <img src="{{ site.baseurl }}/assets/images/forward_pass.png" width="450px"/>
</div>
<br>

### Backward Pass

The backward pass, also known as [**backpropagation**]({{ site.baseurl }}{% link docs/backpropagation.md %}), is where the gradients of the loss function with respect to each parameter (weight and bias) are computed. These gradients guide the updates made to the parameters during gradient descent. The steps involved in the backward pass are:

- **Gradient Calculation**: Compute the gradient of the loss $$\ell$$ with respect to the parameters $$\boldsymbol{w}$$.
- **Parameter Update**: Update the parameters using the gradients. The parameters are adjusted in the opposite direction of the gradient to minimize the loss function.

The gradients are calculated using techniques like the chain rule of calculus, efficiently propagating errors backward through the layers of the model.

<br>
<div align="center">
      <img src="{{ site.baseurl }}/assets/images/backward_pass.png" width="500px"/>
</div>
<br>

## Expected Knowledge

From this text, you should understand the following concepts:

- **Loss Function**: The definition and purpose of the loss function in measuring the classifier's performance and guiding parameter updates.
- **Gradient Descent**: An iterative method to minimize the loss function, crucial for training classifiers.
- **Forward Pass**: The process of computing the loss given the input and true label using the current weights.
- **Backward Pass**: The process of updating the weights using the gradient of the loss function to reduce the loss, effectively implementing gradient descent.
