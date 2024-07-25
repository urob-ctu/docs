---
title: Multi-Layer Perceptron
layout: default
nav_order: 10
parent: Classification
---

# Multi-Layer Perceptron (MLP)
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

Neural networks have a long history but have recently gained immense popularity due to advances in computational power and the availability of large datasets. Artificial Neural Networks (ANNs) are models inspired by the human brain's complexity.

These networks consist of **artificial neurons** or **perceptrons**. Each neuron acts as a computational unit, processing inputs, performing computations, and passing the results to other neurons.

## Artificial Neuron

Understanding artificial neural networks starts with the basic building block: the artificial neuron, or perceptron.

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/perceptron.png" width="800">
</div>

An artificial neuron performs a linear operation followed by a non-linear function called an **activation function**. While various activation functions exist, we'll use the sigmoid function, denoted as $\sigma$, for illustration. Essentially, the neuron performs a non-linear mapping:

$$g: \mathbb{R}^{d} \rightarrow \mathbb{R}$$

$$g(\boldsymbol{x}) = \sigma(\boldsymbol{w} \cdot \boldsymbol{x} + b), \quad \boldsymbol{x}, \boldsymbol{w} \in \mathbb{R}^{d},\;\;b \in \mathbb{R}$$

## Multi-Layer Perceptron

Now, let's delve deeper into the Multi-Layer Perceptron (MLP), the fundamental form of neural networks.

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/mlp.png" width="800">
</div>

The Multi-Layer Perceptron (MLP), or Fully Connected Neural Network, consists of multiple layers of artificial neurons. Each layer processes the input data collectively, with each neuron connected to every neuron in the previous and next layers. This interconnected structure facilitates the propagation of information across the network, enabling it to discern intricate patterns and relationships in the data.

To understand MLP operationally, envision it as a sequence of matrix multiplications and activation functions applied to the input data as it progresses through the layers. This matrix-based approach not only enhances computational efficiency but also offers insight into how data is processed within the network.

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/mlp_vectorized.png" width="800">
</div>

The image illustrates a neural network with two hidden layers, enhancing the model's complexity. This complexity is expressed through a sequence of transformations:

1. **First Hidden Layer:** The output of the first hidden layer, denoted as $\boldsymbol{g}_1(\boldsymbol{x})$, is calculated as follows:
$$\boldsymbol{g}_1(\boldsymbol{x}) = \boldsymbol{\sigma}(\boldsymbol{W}^{(1)} \cdot \boldsymbol{x} + \boldsymbol{b}^{(1)})$$
Where the $\boldsymbol{W}^{(1)}$ is the weight matrix of dimensions $\mathbb{R}^{h_1 \times d}$ and the $\boldsymbol{b}^{(1)}$ is the bias vector of dimensions $\mathbb{R}^{h_1}$

2. **Second Hidden Layer:** Likewise, the output of the second hidden layer, $\boldsymbol{g}_2(\boldsymbol{x})$, is computed as:
$$\boldsymbol{g}_2(\boldsymbol{x}) = \boldsymbol{\sigma}(\boldsymbol{W}^{(2)} \cdot \boldsymbol{x} + \boldsymbol{b}^{(2)})$$ 
Where the $\boldsymbol{W}^{(2)}$ is the weight matrix of dimensions $\mathbb{R}^{h_2 \times h_1}$ and the $\boldsymbol{b}^{(2)}$ is the bias vector of dimensions $\mathbb{R}^{h_2}$

3. **Output Layer:** Finally, the output of the network, $\boldsymbol{y}$, is calculated as:
$$\boldsymbol{g}_3(\boldsymbol{x}) = \boldsymbol{W}^{(3)} \cdot \boldsymbol{x} + \boldsymbol{b}^{(3)}$$
Where the $\boldsymbol{W}^{(3)}$ is the weight matrix of dimensions $\mathbb{R}^{c \times h_2}$ and the $\boldsymbol{b}^{(3)}$ is the bias vector of dimensions $\mathbb{R}^{c}$

In this notation, each hidden layer progressively transforms the input, generating a more abstract representation of the data. This leads us to the **forward pass** of the neural network with two hidden layers, described as:

$$\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{g}_3(\boldsymbol{g}_2(\boldsymbol{g}_1(\boldsymbol{x}))) = \boldsymbol{s}, \quad \boldsymbol{x} \in \mathbb{R}^{d},\; \boldsymbol{s} \in \mathbb{R}^{c}$$

Where $$\boldsymbol{s}$$ refers to the logits of the neural network.

## Under the Hood of Neural Networks

To understand neural networks better, we can break them down into two main components. The first part involves a non-linear transformation of the feature space $\mathbb{R}^{d}$ into a higher-dimensional space, enhancing the data's separability. The second part functions as a linear classifier, akin to what we discussed in the [Linear Classifier section]({{ site.baseurl }}{% link docs/classification/linear_classifier.md %}).

The behavior of the first part heavily relies on the activation function chosen. We'll illustrate this using the example of the spiral dataset, initially not linearly separable. The following videos demonstrate how different activation functions affect network behavior."

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/activation_functions.png" width="800">
</div>

### ReLU Activation Function

The ReLU function is defined as:

$$f(x) = \max(0, x)$$

The ReLU function is the most commonly used activation function in neural networks. We can see that it transforms the feature space only to the positive values.

<div align="center">
<video src="{{ site.baseurl }}/assets/videos/spirals_relu.mp4" width="640" autoplay loop controls muted></video>
</div>


### Sigmoid Activation Function

The sigmoid function is defined as:

$$f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$$

The sigmoid function used to be popular activation function at the beginning of neural networks. However, it has some drawbacks. It squashes the input to the range $$[0, 1]$$ and it has a vanishing gradient problem.

<div align="center">
<video src="{{ site.baseurl }}/assets/videos/spirals_sigmoid.mp4" width="640" autoplay loop controls muted></video>
</div>


### Tanh Activation Function

The Tanh function is defined as:

$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

The Tanh function is another activation function. The main difference between the Tanh and ReLU functions is that Tanh squashes the input to the range $$[-1, 1]$$ which is 0 centered. It also has problems with vanishing gradients.

<div align="center">
<video src="{{ site.baseurl }}/assets/videos/spirals_tanh.mp4" width="640" autoplay loop controls muted></video>
</div>