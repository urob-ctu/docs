---
title: k-NN
layout: default
nav_order: 1
parent: Classification
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

## Expected Knowledge

From this text, you should understand the following concepts:

- **Nearest Neighbors Classifier**: The basic idea and steps of the Nearest Neighbors classifier.
- **Euclidean Distance**: How to calculate the Euclidean distance between two samples.
- **k-Nearest Neighbors Classifier**: The concept of k-Nearest Neighbors and how it differs from the basic Nearest Neighbors classifier.
