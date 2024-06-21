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

Classification is a fundamental concept in machine learning where the goal is to categorize data into predefined classes or labels. For example, an email might be classified as "spam" or "not spam," and a medical image could be classified as showing "cancerous" or "non-cancerous" tissue.


## Dataset

In supervised machine learning, a typical dataset consists of two main components:

1. **Features** - These are the characteristics that describe the data. Usually, a sample is represented as a vector of features, denoted as $\boldsymbol{x} = (x_{1}, x_{2}, ..., x_{d})$. In this assignment, the number of features will be referred to as $d$.

2. **Labels** - These are the values we aim to predict. In this assignment, we'll use $y$ to denote the labels.

The role of the classifier is to take a feature vector $\boldsymbol{x}$ and predict the corresponding label $y$. Mathematically, this can be represented as a function $f$ that maps features to labels:

$$f: \mathbb{R}^{d} \rightarrow \mathcal{Y}$$

$$f(\boldsymbol{x}) = y, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathcal{Y}$$

Once you have your dataset with features and labels, it's crucial to divide it into distinct subsets for training, validation, and testing. These subsets serve specific purposes during the model development and evaluation process:

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

## Examples

You have already come across various classifiers during your studies. Here are some examples to refresh your memory.

### First Example: Nearest Neighbors Classifier

The Nearest Neighbors classifier is a straightforward algorithm that operates as follows:

1. **Training** - The classifier essentially memorizes the training dataset. The training set is stored in the classifier and used for prediction.
2. **Prediction** - When making predictions, the classifier identifies the closest sample in the training set to the given input and assigns the label of the closest sample to the input.

There are various methods to determine the closest sample, but for this assignment, we'll utilize the Euclidean distance metric. The Euclidean distance between two samples, denoted as $$\boldsymbol{x}_{1}$$ and $$\boldsymbol{x}_{2}$$, is defined as:

$$d(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}) = \sqrt{\sum_{i=1}^{d} (x_{1i} - x_{2i})^{2}}$$

Here, $\boldsymbol{x}_{1}$ and $$\boldsymbol{x}_{2}$$ represent two samples from the dataset.

### Second Example: k-Nearest Neighbors Classifier

The k-Nearest Neighbors classifier is akin to the Nearest Neighbors classifier, with one key difference. In the k-Nearest Neighbors classifier, instead of considering just the single closest sample, we find the $k$ closest samples from the training set. The classifier then assigns the label that is most prevalent among these $k$ closest samples to the given input.

To illustrate, here's an example of k-Nearest Neighbors classification with $k=3$ (the yellow point represents the point we want to classify):

<br>
<br>

<div align="center">
      <img src="{{ site.baseurl }}/assets/images/knn_principle.png" alt="k-Nearest Neighbors principle"/>
</div>

<br>
<br>

In this example, the two nearest neighbors belong to the green class, while one belongs to the red class. Consequently, the yellow point is classified as part of the green class.

## Summary

- **What is Classification?**
  - Categorization of data into predefined classes or labels.
  - Mathematically it is a function $f: \mathbb{R}^{d} \rightarrow \mathcal{Y}$.
  - Examples: Email classification (spam/not spam), medical image classification (cancerous/non-cancerous).

- **Dataset Components**
  - **Features**: Characteristics describing the data, represented as a vector $\boldsymbol{x}$.
  - **Labels**: Values to be predicted, denoted as $y$.

- **Dataset Splitting**
  - **Training Set**: Largest portion, used to train the model.
  - **Validation Set**: Used for hyperparameter tuning and model selection.
  - **Test Set**: Independent set for final model evaluation to assess generalization.
  - **Typical Ratios**: 70-80% training, 10-15% validation, 10-15% testing.

- **Basic Classifiers**
  - **Nearest Neighbors Classifier**
    - Memorizes training data.
    - Predicts based on the closest sample using Euclidean distance.
  - **k-Nearest Neighbors Classifier**
    - Considers the $k$ closest samples for prediction.
    - Assigns the most prevalent label among the $k$ nearest samples.