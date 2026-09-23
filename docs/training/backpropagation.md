---
title: Backpropagation
layout: default
nav_order: 3
parent: Training
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

Conceptually, backpropagation is an algorithm for assigning "blame" or "responsibility" for the final loss to every parameter in the network. It starts with the total error at the end and works backward, using the chain rule to calculate how much each weight contributed to that error. Mathematically it is nothing more than the chain rule, organized so that the gradient with respect to *all* parameters is obtained at a cost comparable to one evaluation of the network.

In this text, we will explain the backpropagation algorithm and its components on a small worked example, and then show how the same idea scales to whole layers.

{: .note }
>Note that we will not explain every term used in this text. We assume that you are familiar with the basic concepts of neural networks. If you are not yet, we recommend reading the previous texts in this course.

## The Computation Graph

A computation graph is a directed acyclic graph in which the nodes are elementary operations (multiplication, summation, an activation function, the loss) and the edges carry the values flowing between them. Any neural network can be drawn this way. The graph is simple for a single neuron and large for a deep network, but it is always built from the same few kinds of nodes.

The graph is more than a drawing. Automatic differentiation libraries build exactly this structure during every forward pass so that they can walk it backwards afterwards.

For example, a simple neuron with $$n$$ inputs and one output can be represented in the following way:

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/neuron_graph.png" width="800">
</div>

We can easily visualize the forward and backward pass of the network using the computation graph.

## Components of the Backpropagation Algorithm

One training step consists of three parts: the forward pass, the backward pass and the weight update. Strictly speaking, only the backward pass *is* backpropagation. The forward pass is a prerequisite, because it produces the values the backward pass needs, and the update is the job of the optimizer described on the [gradient descent]({{ site.baseurl }}{% link docs/training/gradient-descent.md %}) page. We go through all three because they only make sense together.

We will use this simple network as a running example: a single neuron with two inputs, no activation function and a squared-error loss.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network.png" width="800">
</div>

$$ z = w_1 x_1 + w_2 x_2 + b, \qquad \mathcal{L} = \tfrac{1}{2}\,(z - z^{*})^2 $$

The factor $$\tfrac{1}{2}$$ is there only to make the derivative tidy, $$\partial\mathcal{L}/\partial z = z - z^{*}$$. The concrete numbers are

$$ \boldsymbol{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.7 \end{bmatrix}, \qquad \begin{bmatrix} w_1 \\ w_2 \\ b \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \\ 1.2 \end{bmatrix}, \qquad z^{*} = 3.5. $$

### The Forward Pass

In the forward pass, the input data is passed through the network, and the output is calculated. The output is then compared to the true (expected) output, and the error is calculated using a loss function. The loss function is a measure of how far the predicted output is from the true output.

In the computation graph the forward pass follows the arrows from left to right. For our example

$$ y_1 = w_1 x_1 = 0.5, \qquad y_2 = w_2 x_2 = 1.4, \qquad z = y_1 + y_2 + b = 3.1, \qquad \mathcal{L} = \tfrac{1}{2}\,(3.1 - 3.5)^2 = 0.08. $$

Every intermediate value ($$y_1$$, $$y_2$$, $$z$$) is kept. The backward pass will need them.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network_fwd.png" width="800">
</div>

### The Backward Pass

In the backward pass, the gradient of the loss function with respect to every parameter of the network is calculated. We walk the graph from the loss back to the inputs, against the arrows, and apply the chain rule at every node.

{: .definition }
>**Chain Rule** is a formula that expresses the derivative of the composition of the differentiable functions $$f$$ and $$g$$ in terms of the derivatives of $$f$$ and $$g$$. More precisely, if $$h = f \circ g$$ is the function such that $$h(x) = f(g(x))$$ for every $$x$$, then the chain rule is, in Lagrange's notation,
>
>$$h'(x) = f'(g(x))\, g'(x),$$
>
>and in Leibniz's notation (if $$y = f(z)$$ and $$z = g(x)$$)
>
>$$\frac{dy}{dx} = \frac{dy}{dz}\,\frac{dz}{dx}.$$

{: .definition }
>**Chain rule with several paths.** If $$x$$ influences the loss through several intermediate variables $$u_1, \dots, u_n$$, the contributions of all paths are summed:
>
>$$\frac{\partial\mathcal{L}}{\partial x} = \sum_{i=1}^{n} \frac{\partial\mathcal{L}}{\partial u_i}\,\frac{\partial u_i}{\partial x}.$$
>
>In the graph this is the rule for a value that fans out into several nodes: the gradients arriving along the different edges are added up. You will need it wherever a parameter is used more than once, for instance in a [convolutional layer]({{ site.baseurl }}{% link docs/models/convolutional-networks.md %}), where one kernel weight affects every output pixel.

Applied to a computation graph, the chain rule turns into a single local rule that is executed once per node:

{: .important }
>**Gradient with respect to an input of a node**
>
>downstream gradient = upstream gradient × local derivative.
>
>Each node receives the gradient of the loss with respect to its own output (the *upstream gradient*, arriving from the right), multiplies it by the derivative of its own operation with respect to each of its inputs, and passes the results (the *downstream gradient*) on to the left. No node ever needs to know what the rest of the network looks like.

Let us execute this rule on the example. Our graph has only three kinds of nodes:

- **Loss node** $$\mathcal{L} = \tfrac{1}{2}(z - z^{*})^2$$. The recursion starts with $$\partial\mathcal{L}/\partial\mathcal{L} = 1$$. The local derivative is $$z - z^{*}$$, so $$\partial\mathcal{L}/\partial z = 1 \cdot (3.1 - 3.5) = -0.4$$.
- **Sum node** $$z = y_1 + y_2 + b$$. The local derivative with respect to each input is $$1$$, so a sum node simply *copies* the upstream gradient to all of its inputs: $$\partial\mathcal{L}/\partial y_1 = \partial\mathcal{L}/\partial y_2 = \partial\mathcal{L}/\partial b = -0.4$$.
- **Product node** $$y_1 = w_1 x_1$$. The local derivative with respect to $$w_1$$ is the *other* input, $$x_1$$, so a product node *swaps* its inputs: $$\partial\mathcal{L}/\partial w_1 = -0.4 \cdot 0.5 = -0.2$$, and likewise $$\partial\mathcal{L}/\partial w_2 = -0.4 \cdot 0.7 = -0.28$$.

This is why the forward values have to be stored: the product node needs $$x_1$$ from the forward pass to compute the gradient for $$w_1$$. The inputs receive gradients as well ($$\partial\mathcal{L}/\partial x_1 = -0.4 \cdot 1.0 = -0.4$$). They are of no use here, but in a deeper network they are exactly the upstream gradient that the previous layer continues with.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network_bwd.png" width="800">
</div>

### Updating the Weights

After the gradients are calculated, the optimizer updates the parameters. The simplest optimizer, gradient descent, moves every parameter a small step against its gradient,

$$ w \leftarrow w - \alpha\,\frac{\partial\mathcal{L}}{\partial w}, $$

where $$\alpha$$ is the learning rate. Stochastic and mini-batch variants, and optimizers such as Adam or RMSprop, differ only in how this step is scaled; see the [gradient descent]({{ site.baseurl }}{% link docs/training/gradient-descent.md %}) page and [Ruder's overview of gradient descent optimizers](https://www.ruder.io/optimizing-gradient-descent/). With plain gradient descent and $$\alpha = 1.0$$ our example becomes

$$ w_1 = 1.0 - 1.0 \cdot (-0.2) = 1.2, \qquad w_2 = 2.0 - 1.0 \cdot (-0.28) = 2.28, \qquad b = 1.2 - 1.0 \cdot (-0.4) = 1.6. $$

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/simple_network_update.png" width="800">
</div>

A quick check that the step went in the right direction: a second forward pass gives $$z = 0.5 \cdot 1.2 + 0.7 \cdot 2.28 + 1.6 = 3.796$$ and $$\mathcal{L} \approx 0.044$$, down from $$0.08$$. The step overshot the target $$z^{*} = 3.5$$ indicating that the learning rate is too large, which is exactly the effect discussed on the gradient descent page.

## Vector-Jacobian Product

For a single neuron we could write out every partial derivative by hand. Real layers map vectors to vectors, $$f : \mathbb{R}^n \rightarrow \mathbb{R}^m$$, and the derivative of such a function is the $$m \times n$$ Jacobian matrix $$\partial f / \partial \boldsymbol{z}$$. The local rule from the previous section still holds, it just becomes a matrix product: the downstream gradient is the upstream gradient multiplied by the Jacobian.

{: .definition }
>The **vector-Jacobian product** (VJP) of a function $$f$$ at the point $$\boldsymbol{z}$$ with a vector $$\boldsymbol{v}$$ is defined as the following snippet (note that we mix pseudocode with mathematical expressions)
>
>def $$\textrm{vjp}_{f}(\boldsymbol{v}, \boldsymbol{z}):$$
>
>&nbsp;&nbsp;&nbsp;&nbsp;return $$\boldsymbol{v}^{\top}\frac{\partial f}{\partial\boldsymbol{z}}$$
>
>In backpropagation $$\boldsymbol{v}$$ is always the upstream gradient $$\partial\mathcal{L}/\partial\boldsymbol{y}$$ of the output $$\boldsymbol{y} = f(\boldsymbol{z})$$. The result is $$\partial\mathcal{L}/\partial\boldsymbol{z}$$: it has the same shape as $$\boldsymbol{z}$$ and becomes the upstream gradient of the node before $$f$$.

The primary advantage of the VJP approach is its efficiency. The Jacobian is never formed. For a linear layer with 1000 inputs and 1000 outputs it would have $$10^6$$ entries per sample, and the Jacobian with respect to the weights of a whole network would be far too large to store, yet the product $$\boldsymbol{v}^{\top}\,\partial f/\partial\boldsymbol{z}$$ can usually be computed as cheaply as the forward pass. Each type of layer therefore implements its own VJP directly, as the following example shows.

{: .note }
>**Example: linear layer** $$\boldsymbol{y} = W\boldsymbol{x} + \boldsymbol{b}$$ with $$W \in \mathbb{R}^{m \times n}$$ (vectors are columns). With the upstream gradient $$\boldsymbol{v} = \partial\mathcal{L}/\partial\boldsymbol{y} \in \mathbb{R}^{m}$$ the three VJPs of the layer are
>
>$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{x}} = W^{\top}\boldsymbol{v}, \qquad \frac{\partial\mathcal{L}}{\partial W} = \boldsymbol{v}\,\boldsymbol{x}^{\top}, \qquad \frac{\partial\mathcal{L}}{\partial\boldsymbol{b}} = \boldsymbol{v}.$$
>
>Only a matrix-vector product and an outer product are needed; the $$m \times mn$$ Jacobian with respect to $$W$$ never appears. Our neuron is the case $$m = 1$$: $$W = [\,w_1 \; w_2\,]$$, $$\boldsymbol{v} = -0.4$$, and $$\boldsymbol{v}\,\boldsymbol{x}^{\top} = [\,-0.2 \;\; -0.28\,]$$ are exactly the gradients from the figure above.

Multiplying the Jacobians from the loss backwards, one node at a time, is called *reverse-mode* automatic differentiation, and it is what makes backpropagation cheap: for a scalar loss, the gradients with respect to millions of parameters are obtained with a single backward pass whose cost is roughly that of one forward pass. Multiplying from the inputs forwards (*forward mode*, the Jacobian-vector product) would need one pass per input dimension instead.

The figure below shows the backward pass through the chain $$\boldsymbol{x} \rightarrow f \rightarrow \boldsymbol{y} \rightarrow g \rightarrow \boldsymbol{z} \rightarrow \mathcal{L}$$ as a sequence of VJPs. It starts with $$\textrm{vjp}_{\mathcal{L}}(1, \boldsymbol{z})$$, and the result of each call is the $$\boldsymbol{v}$$ of the next one: $$\textrm{vjp}_{g}(\partial\mathcal{L}/\partial\boldsymbol{z}, \boldsymbol{y})$$, then $$\textrm{vjp}_{f}(\partial\mathcal{L}/\partial\boldsymbol{y}, \boldsymbol{x})$$.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/vjp_graph.png" width="800">
</div>

## Backpropagation in Practice

In practice you never write VJPs yourself. Automatic differentiation libraries such as PyTorch, JAX or TensorFlow record the computation graph during the forward pass and, when you call `loss.backward()`, they call the VJP of every recorded operation in reverse order. The whole example above is a few lines:

```python
import torch

x = torch.tensor([0.5, 0.7])
w = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor(1.2, requires_grad=True)
z_star = 3.5

z = w @ x + b                      # forward pass, the graph is recorded
loss = 0.5 * (z - z_star) ** 2     # tensor(0.0800)

loss.backward()                    # backward pass, fills .grad
print(w.grad, b.grad)              # tensor([-0.2000, -0.2800]) tensor(-0.4000)

with torch.no_grad():              # gradient descent step with alpha = 1.0
    w -= 1.0 * w.grad
    b -= 1.0 * b.grad
print(w, b)                        # tensor([1.2000, 2.2800]) tensor(1.6000)
```

Understanding what happens underneath is still essential. Two consequences you will meet immediately:

- **Memory.** Every intermediate value has to be kept until the backward pass has consumed it: $$y_1$$, $$y_2$$ and $$z$$ in the example, whole activation tensors for every layer and every sample of the batch in a real network. This is why training needs several times more GPU memory than inference, why the batch size is limited by memory, and why inference code is wrapped in `torch.no_grad()`, which tells the library not to record the graph.
- **Gradient checking.** Any gradient can be verified against a finite difference, $$\partial\mathcal{L}/\partial w \approx \big(\mathcal{L}(w + \varepsilon) - \mathcal{L}(w - \varepsilon)\big) / 2\varepsilon$$ with $$\varepsilon \approx 10^{-5}$$ in double precision. This costs two loss evaluations per parameter, far too slow for training, but it is the standard way to test a hand-written backward pass of a custom layer.

## Expected Knowledge

Answer the following questions to test your understanding of backpropagation.

1. **Conceptual Understanding:** Explain backpropagation in non-mathematical terms. What is a "computation graph," and how does backpropagation use it to compute gradients?

2. **The Chain Rule:** Imagine a simple computation: $$z = w \cdot x + b$$, followed by $$a = \text{ReLU}(z)$$, and a loss $$\mathcal{L} = (a_{true} - a)^2$$. Using the chain rule, write down the expression for the partial derivative of the loss with respect to the weight, $$\frac{\partial \mathcal{L}}{\partial w}$$.

3. **Vector-Jacobian Product (VJP):** What is the main practical advantage of using the Vector-Jacobian Product (VJP) to implement backpropagation in deep learning libraries like PyTorch or TensorFlow, compared to explicitly calculating the entire Jacobian matrix?

4. **Forward vs. Backward Pass:** What key quantity is computed during the **forward pass**? What key quantity is computed for each parameter during the **backward pass**? How do these two passes work together in the context of gradient descent?

5. **Weight Sharing:** The same weight $$w$$ is used twice in a graph, $$z = w x_1 + w x_2$$. Express $$\frac{\partial \mathcal{L}}{\partial w}$$ in terms of $$\frac{\partial \mathcal{L}}{\partial z}$$. Which rule from this page did you use, and why does the same situation arise in a convolutional layer?

6. **Memory:** Why does training a network need considerably more memory than running it for inference, and what exactly has to be stored? What does `torch.no_grad()` change?
