---
title: k-Nearest Neighbors
layout: default
nav_order: 1
parent: Models
---


# k-Nearest Neighbors Classifier
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

The classification task isn't new to you. You have already come across various classifiers during your studies. You should be at least familiar with the name k-Nearest Neighbors Classifier. In this article, we will refresh your memory so we can introduce new topics on these classifiers and compare them to new classifiers.

## Nearest Neighbors Classifier

The Nearest Neighbors classifier is a straightforward algorithm that operates as follows:

1. **Training** - The classifier essentially memorizes the training dataset. The training set is stored in the classifier and used for prediction.
2. **Inference** - When making predictions, the classifier identifies the closest sample in the training set to the given input and assigns the label of the closest sample to the input.

There are various methods to determine the closest sample, but for this assignment, we'll utilize the Euclidean distance metric. The Euclidean distance between two samples, denoted as $$\boldsymbol{x}_{1}$$ and $$\boldsymbol{x}_{2}$$, is defined as:

$$d(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}) = \sqrt{\sum_{i=1}^{d} (x_{1i} - x_{2i})^{2}}$$

Here, $\boldsymbol{x}_{1}$ and $$\boldsymbol{x}_{2}$$ represent two samples from the dataset.

## k-Nearest Neighbors Classifier

The k-Nearest Neighbors classifier is akin to the Nearest Neighbors classifier, with one key difference. In the k-Nearest Neighbors classifier, instead of considering just the single closest sample, we find the $k$ closest samples from the training set. The classifier then assigns the label that is most prevalent among these $k$ closest samples to the given input.

To illustrate, here's an example of k-Nearest Neighbors classification with $k=3$ (the yellow point represents the point we want to classify):

<br>
<div align="center">
  <figure>
    <img src="{{ site.baseurl }}/assets/images/knn_principle.png" width="300px">
    <!-- <figcaption>KNN Principle</figcaption> -->
  </figure>
</div>
<br>

In this example, the two nearest neighbors belong to the green class, while one belongs to the red class. Consequently, the yellow point is classified as part of the green class.

## Advantages and Disadvantages of k-NN

**Advantages:**

* **Simple and Intuitive:** The algorithm is easy to understand and implement.
* **No Training Phase:** The "training" is just storing the dataset, which is very fast.
* **Adapts to new data easily:** New training samples can be added without retraining the entire model.

**Disadvantages:**

* **Slow Inference:** The algorithm must compute the distance to *every* training sample to make a single prediction. This is computationally expensive, especially with large datasets, making it unsuitable for many real-time robotics applications.
* **Curse of Dimensionality:** Performance degrades as the number of features (dimensions) increases, because the distance between points becomes less meaningful.
* **Sensitive to irrelevant features:** All features contribute to the distance calculation, so irrelevant or noisy features can significantly degrade performance.
* **Requires feature scaling:** Features with larger ranges can dominate the distance calculation, so data normalization (e.g., scaling to [0, 1]) is often necessary.

## Expected Knowledge

Answer the following questions to test your understanding of the k-NN algorithm.

1. **Algorithmic Steps:** A k-NN classifier has already "trained" on a dataset. Describe, step-by-step, the process it follows to classify a new, unseen data point $$\boldsymbol{x}_{new}$$.

2. **Application & Calculation:** In the image provided, if you were to classify the yellow point using $$k=5$$ instead of $$k=3$$, what would the new classification be? Explain your reasoning.

3. **Performance Trade-offs:** The "training" phase for k-NN is extremely fast, while the "inference" phase is very slow. Explain why this is the case. Why might this characteristic make k-NN a poor choice for a self-driving car's object detection system?
