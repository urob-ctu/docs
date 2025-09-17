---
title: Classification
layout: default
has_children: true
nav_order: 2
mathjax: true
---


# Classification
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

Classification is a basic concept in machine learning where we categorize data into predefined classes or labels. For example, an email might be classified as "spam" or "not spam," and a medical image could be classified as showing "cancerous" or "non-cancerous" tissue.

## Definition of Classification

Before we dive into the details of classification, let's introduce **Features** and **Labels**.

1. **Features** - These are the attributes or properties that represent the data. A feature can take different forms, such as a numerical value, a vector, or even an image. In this assignment, we'll use $x$ to represent the features. The collection of all possible features, or the set of all feature values, is called the feature space, and we will denote it by $\mathcal{X}$.

2. **Labels** - These are the values we aim to predict. We'll use $y$ to denote the labels. The set of all possible labels will be referred to as $\mathcal{Y}$. For example, in an image classification problem where we want to predict whether an image is a dog or a cat, $\mathcal{Y}$ would be {dog,cat}.

{: .definition }
>**The Classification** is the task of mapping from a feature space to a label space. The goal is to learn a function $\mathcal{P}$ that maps features to labels. This function is called the **Prediction Function** or **Prediction Strategy** or **Inference Rule**.
>
>$$\mathcal{P}: \mathcal{X} \rightarrow \mathcal{Y}$$
>
>$$\mathcal{P}(x) = y, \quad x \in \mathcal{X}, \quad y \in \mathcal{Y}$$

Although this definition is correct, we will make a few simplifications for our course.

1. We will assume the feature space is a vector space of dimension $d$, where $d$ is the number of features. Therefore, we will denote the feature space as $\mathcal{X} = \mathbb{R}^{d}$ and feature vectors as $\boldsymbol{x} \in \mathbb{R}^{d}$.
2. We will assume the label space is a finite set of natural numbers. Therefore, we will denote the label space as $\mathcal{Y} = \mathbb{N}$.
3. We will also differentiate between the **prediction** and the **label**. We will denote the label as $y \in \mathcal{Y}$ and the prediction as $\hat{y} \in \mathcal{Y}$.

{: .note }
>For this course **Classification** is the process of mapping from a feature space $\mathbb{R}^{d}$ to a label space $\mathbb{N}$. The task is to learn a prediction function $\mathcal{P}$ that creates a prediction $\hat{y}$ from the feature vector $\boldsymbol{x}$ and we want the prediction to be as close as possible to the true label $y$.
>
>$$\mathcal{P}(x) = \hat{y}, \quad x \in \mathbb{R}^{d}, \quad \hat{y} \in \mathbb{N}$$

This can be visualized as the following schematic:

<br>
<div align="center">
    <img src="{{ site.baseurl }}/assets/images/predictor_function.png" width="300px"/>
</div>
<br>

## Dataset

You should already be familiar with the concept of a dataset. But let's define it for our classification task.

{: .definition }
>**Dataset** $$\mathcal{D}$$ is a collection of **samples** $$(\boldsymbol{x}, y)$$ where $$\boldsymbol{x}$$ is a feature vector and $$y$$ is a label.
>
>$$\mathcal{D} = \{(\boldsymbol{x}_{1}, y_{1}), \ldots, (\boldsymbol{x}_{n}, y_{n})\}, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{N}$$

The dataset is used for training, validation, and testing of the model, so it is usually divided into these three subsets.

### Dataset Splits

1. **Training Set** - 
    - **Purpose**: The training set is used to train the machine learning model. The model learns the relationships between the input features and the target labels in this set.
    - **How it's used**: During training, the algorithm iteratively adjusts its parameters (like weights in neural networks) to minimize the error (or loss) in predicting the target variable based on the input features.
    - **Goal**: To enable the model to generalize from the data, learning patterns and relationships in the input data that help make accurate predictions.
2. **Validation Set** -
    - **Purpose**: The validation set is used to fine-tune the model and to perform hyperparameter tuning (like selecting the learning rate, number of layers, etc.). It's also used to prevent overfitting (when a model performs well on training data but poorly on unseen data).
    - **How it's used**: The model’s performance on the validation set is evaluated after each training epoch. Based on this evaluation, adjustments can be made to improve the model.
    - **Goal**: To provide an unbiased evaluation during training and help select the best model configuration without exposing the model to the test set.
3. **Test Set** -
    - **Purpose**: The test set is used for the final evaluation of the model. This set contains data that the model has never seen during training or validation.
    - **How it's used**: After training and fine-tuning, the test set is used to evaluate the model’s generalization ability on completely new data.
    - **Goal**: To provide an objective measure of the model’s performance in a real-world setting, ensuring that it can generalize well to unseen data.

{: .note } 
In some situations, the terms "validation set" and "test set" may be used interchangeably. For example, in online literature, the validation set is often referred to as the test set. Additionally, in certain research papers, there may not be a distinct test set, and model evaluation is performed solely on the validation set. However, for this assignment, we will treat the validation and testing sets as separate entities.*

Typical ratios for splitting the dataset include allocating 70-80% for training, 10-15% for validation, and another 10-15% for testing. However, these ratios can vary depending on the dataset size and the specific problem you're addressing.

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/splits.png" width="800px" />
    <hr>
</div>

## Expected Knowledge

From this text, you should understand the following concepts:

- **Features** and **Labels**: What are they and some examples.
- **Prediction Function**: The definition of the prediction function.
- **Dataset splits**: The training, validation, and test sets, and their roles in model development. 
