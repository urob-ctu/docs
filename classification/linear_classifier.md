---
title: Linear Classifier
layout: default
nav_order: 3
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

## Introduction

While the k-NN classifier offers a straightforward concept, it comes with a significant computational cost during testing. This, coupled with its suboptimal performance in many real-world scenarios, underscores the necessity for a more efficient classifier. In this section, we will delve into a more sophisticated type of classifier: the linear classifier. Although the linear classifier is still fairly simple, it is a foundational block in neural networks and more advanced machine learning models.

## Definition of Linear Classifier

In the k-NN classifier, the logits or scores $$\boldsymbol{s}$$ are calculated based on the number of nearest neighbors from each class. Specifically:

$$
\boldsymbol{s} = \begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_c \end{bmatrix}, \quad s_i = \text{num_nearest_neighbors}(\boldsymbol{x}) \text{ from class } i
$$

The linear classifier differs primarily in how it calculates the logits. Instead of counting neighbors, the linear classifier uses a linear function to compute these scores.

{: .definition }
> A **Linear Classifier** is a classifier that uses a linear function to compute the logits. The linear function is defined as follows:
>
> $$
> \boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{s} = \boldsymbol{W} \boldsymbol{x} + \boldsymbol{b}, \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad \boldsymbol{W} \in \mathbb{R}^{c \times d}, \quad \boldsymbol{b} \in \mathbb{R}^{c}
> $$
>
> Here, $\boldsymbol{W}$ denotes a **weight matrix** of size $c \times d$, and $\boldsymbol{b}$ is a **bias vector** of size $c$. These two components, the weights matrix and the bias vector, constitute the classifier's parameters, which are learned from the training data. In linear algebra, this linear function is often referred to as an affine function.

## Training and Inference

- **Training**: The process of training a linear classifier involves finding the optimal weight matrix $$\boldsymbol{W}$$ and bias vector $$\boldsymbol{b}$$. This is typically done using a method called **gradient descent**, which will be discussed in detail in the next section on how to [train classifiers]({{ site.baseurl }}{% link classification/training.md %}).
- **Inference**: Once the model is trained, inference refers to the process of making a prediction for a given input $$\boldsymbol{x}$$. The prediction of the linear classifier is made by finding the class corresponding to the maximum value of the logits:
  
  $$
  \hat{y} = \arg\max (\boldsymbol{W} \boldsymbol{x} + \boldsymbol{b}).
  $$

## Properties of Linear Classifier

The most significant advantage of the linear classifier is its **simplicity**. Linear functions are among the simplest functions in mathematics, making them easy to understand and work with.

However, when we visualize the decision boundaries of a linear classifier, we see that these boundaries are straight lines (or hyperplanes in higher-dimensional spaces) that separate the classes. This means that linear classifiers perform well only if the data is **linearly separable**.

<br>
<div align="center">
      <img src="{{ site.baseurl }}/assets/images/linear_classifier.png" width="800px" />
</div>
<br>

This requirement for linear separability is the main drawback of linear classifiers. In many real-world datasets, the classes are not linearly separable, which limits the effectiveness of linear classifiers.

## Linear Classifier as Template Matching Algorithm

When we classify images using a linear classifier, it is helpful to think of the classifier as performing a form of template matching. Let's explore this concept with an example.

Imagine we have images of animals, and each image contains a dog, cat, or bird. We want to classify each image into one of these three classes. Using a linear classifier, we transform each image into a vector $$ \boldsymbol{x} $$ of size $$d = \text{width} \times \text{height}$$. During inference, we multiply $$ \boldsymbol{x} $$ by the weight matrix $$\boldsymbol{W}$$ and add the bias vector $$\boldsymbol{b}$$ to get the logits $$\boldsymbol{s}$$.

<br>
<div align="center">
      <img src="{{ site.baseurl }}/assets/images/linear_classifier_template_principle.png" width="800px" />
</div>
<br>

- **Matrix Multiplication**: The vector $$\boldsymbol{x}$$ is multiplied by the weight matrix $$\boldsymbol{W}$$. Each row of $$\boldsymbol{W}$$ acts as a template for one of the classes (dog, cat, bird).

    - *Templates in $$\boldsymbol{W}$$*: The weight matrix $$\boldsymbol{W}$$ consists of rows where each row represents a class. For instance, if we have three classes (dog, cat, bird), $$\boldsymbol{W}$$ will have three rows. Each row contains weights that capture the features of that specific class. These weights act like a template or pattern that the classifier uses to recognize the corresponding class in an input image.

    - *Dot Product Interpretation*: When we multiply the input vector $$\boldsymbol{x}$$ with the weight matrix $$\boldsymbol{W}$$, we are essentially computing the dot product between $$\boldsymbol{x}$$ and each row of $$\boldsymbol{W}$$. The dot product measures similarity. A higher dot product value indicates greater similarity between the input image and the class template. For example, if the input image closely resembles the template for "cat," the dot product with the "cat" row in $$\boldsymbol{W}$$ will be higher than with the "dog" or "bird" rows.

- **Adding Bias**: The bias vector $$\boldsymbol{b}$$ is added to the result of the matrix multiplication. The bias vector serves an important role in adjusting the scores. Each element in $$\boldsymbol{b}$$ corresponds to a class and shifts the score for that class up or down. This adjustment allows the classifier to fine-tune its confidence levels for each class independently. For instance, if the classifier is generally biased towards predicting "cat" too often, the bias value for the "cat" class can be reduced to compensate. Conversely, if the classifier tends to underpredict "bird," the bias value for the "bird" class can be increased.

### Example of Templates in CIFAR-10 Dataset

The CIFAR-10 dataset is a typical image dataset used for classification tasks. It consists of 60,000 32x32 color images distributed across 10 distinct classes, with each class containing 6,000 images. The 10 classes are:

<br>
<div align="center">
      <img src="{{ site.baseurl }}/assets/images/cifar-10.png" width="600px" />
</div>
<br>

We trained a linear classifier on this dataset and visualized the rows of the weight matrix $$\boldsymbol{W}$$ as images. This shows us what we mean by templates in $$\boldsymbol{W}$$:

<br>
<div align="center">
      <img src="{{ site.baseurl }}/assets/images/templates.png" width="800px" />
</div>
<br>

For example, if we look at the *plane* or *ship* templates, we see mostly blue backgrounds. This is because planes are typically seen in the sky and ships in the water. We also see a template of a green *frog* in the middle of the image.

We have mentioned the training of classifiers multiple times. Let's now proceed to the next section, which covers how to [train classifiers]({{ site.baseurl }}{% link classification/training.md %}).

## Expected Knowledge

From this text, you should understand the following concepts:

- **Linear Classifier**: What it is and how it differs from the k-NN classifier.
- **Training and Inference**: The processes of training and making predictions with a linear classifier.
- **Properties of Linear Classifier**: The advantages and limitations..
- **Template Matching Algorithm**: Understanding the linear classifier as a template matching algorithm, particularly when applied to image classification.