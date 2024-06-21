---
title: Linear Classifier
layout: default
nav_order: 1
parent: Classification
---

# Linear Classifier
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

While the k-NN classifier offers a straightforward concept, it comes with a computational cost during testing. Coupled with its suboptimal performance in real-world scenarios, this underscores the necessity for a more efficient classifier. Now we will delve into more complicated classifier - linear classifier. Although the linear classifier if still fairly simple it is a foundational block in neural networks. 

## Score Function

In our beloved k-NN classifier, the score function is defined as follows:

$$ \boldsymbol{s} = \begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_c \end{bmatrix}, \quad s_i = \text{num_nearest_neighbors}(\boldsymbol{x}) \text{ from class } i $$

We will see that the linear classifier is different only in calculating the score vector. Maybe you would't be surprised with the fact that the score function is as the name suggests a linear function.

$$\boldsymbol{g}(\boldsymbol{x}) = \boldsymbol{s} = \boldsymbol{W} \boldsymbol{x} + \boldsymbol{b}, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad \boldsymbol{W} \in \mathbb{R}^{c \times d}, \quad \boldsymbol{b} \in \mathbb{R}^{c}$$

Here, $\boldsymbol{W}$ denotes a **weight matrix** of size $c \times d$, and $\boldsymbol{b}$ is a **bias vector** of size $c$. These two components, the weights matrix and the bias vector, constitute the classifier's parameters. We will learn these parameters through training data. It's also worth noting that the linear function is often referred to as the affine function in linear algebra.

## Training and inference

- **Training** is the process of finding the optimal weight matrix $$\boldsymbol{W}$$ and bias vector $$\boldsymbol{b}$$. We will learn these parameters through training data with a procedure called **gradient descent**.
- **Inference** is the process of creating a prediction from a given input $$\boldsymbol{x}$$. The output of the linear classifier is $$ \arg\max (\boldsymbol{s} = \boldsymbol{W} \boldsymbol{x} + \boldsymbol{b})$$.

## Properties of Linear Classifier

The biggest plus of the linear classifier lies in its **simplicity**. The linear functions are one of the simplest functions in mathematics. We understand them well and therefore we like them.

But if we visualize the decision boundaries of the linear classifier, we can see that the boundaries are lines that separate the classes. In higher dimensional spaces, the decision boundaries are hyperplanes that separate the classes.

<div align="center">
      <img src="{{ site.baseurl }}/assets/images/linear_classifier.png" width="800px" />
</div>

This is the main drawback of the linear classifiers. To have a good performance, the **data has to be linearly separable**.

## Linear Classifier as Template Matching Algorithm

<div align="center">
      <img src="{{ site.baseurl }}/assets/images/templates.png" width="800px" />
</div>
