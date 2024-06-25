---
title: Logits / Scores
layout: default
nav_order: 2
parent: Classification
---

# Logits / Scores
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

As we previously discussed, a classifier can be viewed as a function that maps features to corresponding labels. This means that for any given input data, the classifier produces a label that indicates the class to which the input belongs.

<br>
<div align="center">
<figure>
      <img src="{{ site.baseurl }}/assets/images/predictor_function.png" width="300px"/>
      <a id="predictor_function"></a>
</figure>
</div>
<br>

While this approach is effective, it can sometimes be limiting. Let's consider a scenario where we use a 100-Nearest Neighbors classifier in a 2-dimensional space ($\mathbb{R}^{2}$). Suppose we have two samples, $\boldsymbol{x}_{1}$ and $\boldsymbol{x}_{2}$, and we want to make predictions for them. We will use a function:

```
   nearest_neighbors(x, k) -> {n1, n2}
```

This function returns $n_1$ and $n_2$, which are the counts of the nearest neighbors from class 1 and class 2, respectively.

1. **For the first sample, $\boldsymbol{x}_{1}$:**
   - We run the `nearest_neighbors` function and get:
     $$n_1 = 51, \quad n_2 = 49$$
   - Since $n_1$ is greater than $n_2$, our prediction is class 1. Therefore, the predicted label ($\hat{y}_1$) is 1.

2. **For the second sample, $\boldsymbol{x}_{2}$:**
   - We run the `nearest_neighbors` function and get:
     $$n_1 = 99, \quad n_2 = 1$$
   - Since $n_1$ is much greater than $n_2$, our prediction is again class 1. Therefore, the predicted label ($\hat{y}_2$) is also 1.

Although both samples are predicted to belong to class 1, we have different levels of confidence in these predictions. For $$\boldsymbol{x}_{1}$$, the counts are close (51 vs. 49), suggesting uncertainty. For $$\boldsymbol{x}_{2}$$, the counts are very different (99 vs. 1), indicating a high level of certainty. However, just looking at the predicted labels, we can't distinguish this difference in confidence. This is a problem because we might want to know how certain our classifier is about its predictions, especially in critical applications where decisions can have significant consequences.

To address this, we introduce the concept of **logits** or **scores** ($$\boldsymbol{s}$$).

## Definition of Logits / Scores

In our example, we can use the counts $n_1$ and $n_2$ as scores that reflect the classifier's confidence. Let's define these scores more formally:

$$
\boldsymbol{s} = \begin{bmatrix} n_1 \\ n_2 \end{bmatrix}
$$

Here, $n_1$ and $n_2$ are the counts of the nearest neighbors from class 1 and class 2, respectively. The scores vector $\boldsymbol{s}$ gives us a way to quantify the confidence of the classifier in each class. When we want to make a prediction based on the logits, we simply choose the class corresponding to the maximum value in $\boldsymbol{s}$:

$$
\hat{y} = \arg\max \boldsymbol{s}
$$

{: .definition }
> The **logits / scores** $\boldsymbol{s}$ is a vector of length $c$, where $c$ is the number of classes. It quantifies the classifier's confidence in each class. The elements of $\boldsymbol{s}$ represent the degree of certainty or score associated with each class, indicating how strongly the classifier leans towards each possible classification outcome. The logits are then used for prediction, where the class with the highest score is the predicted class.

{: .note }
> The terms logits and scores are often used interchangeably, though they can have slightly different connotations depending on the context:
>
>- **Logits**: This term is frequently used in the context of neural networks, particularly in the final layer of a classification model. Logits are the raw, unnormalized scores output by a neural network. These scores are typically transformed into probabilities using a function like softmax.
>- **Scores / Score Vector**: This term is more general and can refer to any vector of scores or confidences that a classifier assigns to different classes. It doesn't necessarily imply that the scores are unnormalized, though it often does. The score vector could be the direct output of a model or the result of some intermediate computation.
>
> In this course, we will primarily use the term **logits**, as it is more commonly used in the context of neural networks.

## Modification of the Prediction Function

With this understanding, we can modify our classification task to include an intermediate step involving logits. Every classifier, whether it's a simple k-Nearest Neighbors or a complex neural network, has some form of logits inside its prediction function. This step is crucial, so we define a new function $\boldsymbol{f}$ that returns the logits $\boldsymbol{s}$:

$$
\boldsymbol{f}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{c}
$$

$$
\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{s}, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad \boldsymbol{s} \in \mathbb{R}^{c}
$$

This function maps a feature vector $\boldsymbol{x}$ to a logits vector $\boldsymbol{s}$. The prediction is then made by selecting the class with the highest logit value. The relationship between the logits $\boldsymbol{s}$ and the class prediction $\hat{y}$ can be visualized as follows:

<br>
<div align="center">
<figure>
      <img src="{{ site.baseurl }}/assets/images/predictor_detailed.png" width="500px"/>
      <a id="predictor_detailed"></a>
      <figcaption><strong>Predictor Function</strong>: This scheme can be applied to any classifier in the inference stage.</figcaption>
</figure>
</div>
<br>

Understanding logits is fundamental as we move forward to more advanced classifiers, such as the [Linear Classifier]({{ site.baseurl }}{% link classification/linear_classifier.md %}).

## Expected Knowledge

From this text, you should understand the following concepts:

- **Logits / Scores**: What they are and how they quantify the classifier's confidence in each class.
- **Modification of Prediction Function**: Scheme of the prediction function involving logits.