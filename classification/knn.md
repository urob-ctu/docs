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

The classification task isn't new to you. You have already came across various classifiers during your studies. You should be at least familiar with the name k-Nearest Neighbors Classifier. In this article, we will refresh your memory and introduce new topics as well.

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
<br>

<div align="center">
      <img src="{{ site.baseurl }}/assets/images/knn_principle.png" alt="k-Nearest Neighbors principle"/>
</div>

<br>
<br>

In this example, the two nearest neighbors belong to the green class, while one belongs to the red class. Consequently, the yellow point is classified as part of the green class.

## Score Vector

As previously mentioned, a classifier can be viewed as a function denoted as $f$, which maps features to corresponding labels. This concept can be expressed as:

$$f: \mathbb{R}^{d} \rightarrow \mathcal{Y}$$

$$f(\boldsymbol{x}) = y, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathcal{Y}$$

This is what we wanted, but now here's the problem. Let's say we have 100-Nearest Neighbors classifier in $$R^{2}$$ and we have two samples $$\boldsymbol{x}_{1}$$ and $$\boldsymbol{x}_{2}$$, and we want to prediction for. We will also have a function 

```
   nearest_neighbors(x, k) -> {n1, n2}
```

where the output $$n_1$$ and $$n_2$$ are the nearest neighbors from class 1, and 2, respectively.

1. We will run the ``nearest_neighbors`` function on $$\boldsymbol{x}_{1}$$ and we will get
   
   $$n_1 = 51, \quad n_2 = 49$$

   therefore **our prediction is class 1** - $$f(\boldsymbol{x}_{1}) = 1$$.

2. We will run the ``nearest_neighbors`` function on $$\boldsymbol{x}_{2}$$ and we will get

   $$n_1 = 99, \quad n_2 = 1$$

   therefore **our prediction is class also 1** - $$f(\boldsymbol{x}_{2}) = 1$$.

Our predictions are the same even though we sense that we are certainly more sure about the second sample $$\boldsymbol{x}_{2}$$ than the first sample $$\boldsymbol{x}_{1}$$. But we can't detect this only from the output! Isn't that frustrating? What if based on our classification some life would be at stake? Because of this (and ton of other reasons), we need a way to measure the quality/uncertainty of our predictions. That's why we introduce the **score vector** $$\boldsymbol{s}$$. The score vector in our case would be

$$\boldsymbol{s} = \begin{bmatrix} n_1 \\ n_2 \end{bmatrix}$$

where $$n_1$$ and $$n_2$$ are the nearest neighbors from class 1 and 2, respectively. When we want to create prediction from the score vector we simply take the index, where is the maximum value in $$\boldsymbol{s}$$.

$$c = \arg\max_{y \in \mathcal{Y}} \boldsymbol{s}_{y}$$

{: .definition }
The **score vector** $\boldsymbol{s}$ is a vector of length $c$, where $c$ is the number of classes. It quantifies the classifier's confidence in each class. The elements of $\boldsymbol{s}$ represent the degree of certainty or score associated with each class, indicating how strongly the classifier leans towards each possible classification outcome. The score vector is then used for prediction, where the class with the highest score is the predicted class.

So now we can modify the classification task a little bit with an intermediate step.

$$\boldsymbol{g}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{c}$$

$$\boldsymbol{g}(\boldsymbol{x}) = \boldsymbol{s}, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad \boldsymbol{s} \in \mathbb{R}^{c}$$

The relationship between the score vector $$\boldsymbol{s}$$ and the class label $$y$$ is following

$$y = \arg\max (\boldsymbol{s}), \quad \boldsymbol{s} = \begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_c \end{bmatrix}, \quad y \in \mathcal{Y} = \{1, 2, \dots, c\}$$

so in other words

$$\boldsymbol{f}(\boldsymbol{x})) = \arg\max (\boldsymbol{g}(\boldsymbol{x}))$$


## Expected Knowledge

- **Nearest Neighbors Classifier**
  - How we train the NN classifier?
  - Describe the prediction process of the NN classifier.
  - Why do we need the distance metric in NN classifier and what is the most common one?

- **k-Nearest Neighbors Classifier**
  - What is the difference between the k-NN and NN classifier?
  - Describe the prediction process of the k-NN classifier.

- **Score Vector**
  - What is the score vector?
  - What is the relationship between the score vector and the class label?
  - What is the advantage of using the score vector?

