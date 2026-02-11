---
title: Datasets and Features
layout: default
nav_order: 1
parent: Fundamentals
mathjax: true
---

# Datasets and Features
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Features and Labels

Before we dive into machine learning algorithms, let's establish the fundamental building blocks: **Features** and **Labels**.

1. **Features** - These are the attributes or properties that represent the data. A feature can take different forms, such as a numerical value, a vector, or even an image. In this course, we'll use $\boldsymbol{x}$ to represent the features. The collection of all possible features, or the set of all feature values, is called the feature space, and we will denote it by $\mathcal{X}$.

2. **Labels** - These are the values we aim to predict. We'll use $y$ to denote the labels. The set of all possible labels will be referred to as $\mathcal{Y}$. For example, in an image classification problem where we want to predict whether an image is a dog or a cat, $\mathcal{Y}$ would be {dog,cat}.

For example, if we want a robotic arm to pick up an object from a table, our **features** ($$\boldsymbol{x}$$) could be data from a camera, such as the object's pixel coordinates, color, and size, combined with readings from a force sensor in the gripper. The **label** ($$y$$) we want to predict might be a category like {'bolt', 'nut', 'washer'}, or a continuous value like the object's weight in grams.

For this course, we will make a few simplifications:

1. We will assume the feature space is a vector space of dimension $d$, where $d$ is the number of features. Therefore, we will denote the feature space as $\mathcal{X} = \mathbb{R}^{d}$ and feature vectors as $\boldsymbol{x} \in \mathbb{R}^{d}$.
2. We will assume the label space is a finite set of natural numbers. Therefore, we will denote the label space as $\mathcal{Y} = \mathbb{N}$.
3. We will also differentiate between the **prediction** and the **label**. We will denote the label as $y \in \mathcal{Y}$ and the prediction as $\hat{y} \in \mathcal{Y}$.

## Dataset

You should already be familiar with the concept of a dataset. But let's define it for our machine learning tasks.

{: .definition }
>**Dataset** $$\mathcal{D}$$ is a collection of **samples** $$(\boldsymbol{x}, y)$$ where $$\boldsymbol{x}$$ is a feature vector and $$y$$ is a label.
>
>$$\mathcal{D} = \{(\boldsymbol{x}_{1}, y_{1}), \ldots, (\boldsymbol{x}_{n}, y_{n})\}, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{N}$$

The dataset is used for training, validation, and testing of the model, so it is usually divided into these three subsets.

## Dataset Splits

1. **Training Set** - 
    - **Purpose**: The training set is used to train the machine learning model. The model learns the relationships between the input features and the target labels in this set.
    - **How it's used**: During training, the algorithm iteratively adjusts its parameters (like weights in neural networks) to minimize the error (or loss) in predicting the target variable based on the input features.
    - **Goal**: To enable the model to generalize from the data, learning patterns and relationships in the input data that help make accurate predictions.
2. **Validation Set** -
    - **Purpose**: The validation set is used to fine-tune the model and to perform hyperparameter tuning (like selecting the learning rate, number of layers, etc.). It's also used to prevent overfitting (when a model performs well on training data but poorly on unseen data).
    - **How it's used**: The model's performance on the validation set is evaluated after each training epoch. Based on this evaluation, adjustments can be made to improve the model.
    - **Goal**: To provide an unbiased evaluation during training and help select the best model configuration without exposing the model to the test set.
3. **Test Set** -
    - **Purpose**: The test set is used for the final evaluation of the model. This set contains data that the model has never seen during training or validation.
    - **How it's used**: After training and fine-tuning, the test set is used to evaluate the model's generalization ability on completely new data.
    - **Goal**: To provide an objective measure of the model's performance in a real-world setting, ensuring that it can generalize well to unseen data.

{: .note } 
In some situations, the terms "validation set" and "test set" may be used interchangeably. For example, in online literature, the validation set is often referred to as the test set. Additionally, in certain research papers, there may not be a distinct test set, and model evaluation is performed solely on the validation set. However, for this course, we will treat the validation and testing sets as separate entities.

Typical ratios for splitting the dataset include allocating 70-80% for training, 10-15% for validation, and another 10-15% for testing. However, these ratios can vary depending on the dataset size and the specific problem you're addressing.

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/splits.png" width="800px" />
    <hr>
</div>

## Expected Knowledge

Answer the following questions to test your understanding of the core concepts of ML data.

1. **Conceptual Application:** Imagine you are building a system for a mobile robot to navigate an office. Its task is to decide whether to `move forward`, `stop`, or `turn`. The robot is equipped with a laser scanner that provides 180 distance readings in front of it. In this context, define what would constitute:
   - The feature space $$\mathcal{X}$$
   - A single feature vector $$\boldsymbol{x}$$
   - The label space $$\mathcal{Y}$$
   - A true label $$y$$ vs. a prediction $$\hat{y}$$

2. **The Role of Data Splits:** Why do we split a dataset into training, validation, *and* test sets? What specific mistake or problem does each split help us prevent? Describe what would likely happen if you trained your model and evaluated its final performance on the same training set.

3. **Feature Representation:** A feature vector $$\boldsymbol{x} \in \mathbb{R}^d$$ is an abstraction. If your input data is a single 100x100 pixel grayscale image, describe one simple way you could transform this image into a feature vector $$\boldsymbol{x}$$. What would the dimension $$d$$ be in your example?