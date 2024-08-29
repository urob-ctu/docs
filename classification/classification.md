---
title: Classification
layout: default
has_children: true
nav_order: 1
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

1. **Features** - These are the characteristics that describe the data. They can be a number, a vector, an image, etc. In this assignment, we'll use $x$ to denote the features. The feature space will be referred to as $\mathcal{X}$.

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

1. **Training Set** - This is the largest portion of your dataset and is used to train the model. The model learns underlying patterns in the data by adjusting its parameters based on the training examples. A larger training set generally helps the model learn more accurate and generalizable patterns.

2. **Validation Set** - This subset is used during training for hyperparameter tuning and model selection. Hyperparameters are settings that are not learned by the model during training, such as the learning rate or the number of hidden layers in a neural network. The model's performance on the validation set helps in selecting the best hyperparameters and prevents overfitting, where the model becomes too tailored to the training data and performs poorly on new, unseen data.

3. **Test Set** - This set is entirely independent of the training and validation data. Its purpose is to evaluate the final model's performance after it has been trained and tuned. Using a separate test set provides an unbiased estimate of how well the model is likely to perform on new, unseen data. This step is essential for assessing the model's generalization capabilities.

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