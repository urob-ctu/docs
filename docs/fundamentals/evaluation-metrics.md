---
title: Classification Metrics
layout: default
nav_order: 4
parent: Fundamentals
---

# Classification Metrics
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

After training a classifier, it's crucial to evaluate its performance to understand how well it generalizes to unseen data. Simply knowing that a model makes predictions isn't enough—we need quantitative measures to assess the quality of those predictions. This section covers the fundamental metrics used to evaluate classification models.

## Confusion Matrix

A confusion matrix is a table that provides a detailed breakdown of correct and incorrect predictions for each class. It's particularly useful for understanding which classes are being confused with each other and forms the foundation for calculating other metrics.

{: .definition }
> **Confusion Matrix** is a square matrix where rows represent the true classes and columns represent the predicted classes. Each cell $(i,j)$ contains the number of samples that belong to class $i$ but were predicted as class $j$.

For a binary classification problem, the confusion matrix looks like:

$$
\begin{array}{c|cc}
 & \text{Predicted} \\
\text{Actual} & \text{Positive} & \text{Negative} \\
\hline
\text{Positive} & TP & FN \\
\text{Negative} & FP & TN \\
\end{array}
$$

where:
- **TP** (True Positives): Correctly predicted positive cases
- **TN** (True Negatives): Correctly predicted negative cases  
- **FP** (False Positives): Incorrectly predicted as positive
- **FN** (False Negatives): Incorrectly predicted as negative

## Accuracy

Accuracy is the most intuitive metric for classification performance. It measures the proportion of correct predictions among all predictions made.

{: .definition }
> **Accuracy** is defined as the ratio of correctly predicted samples to the total number of samples:
>
> $$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

While accuracy is easy to understand and compute, it can be misleading in cases of **class imbalance**. For example, if 95% of emails are not spam, a classifier that always predicts "not spam" would achieve 95% accuracy but would be useless for detecting actual spam.

## Precision, Recall, and F1-Score

For binary classification, particularly when dealing with imbalanced datasets, precision and recall provide more nuanced evaluation metrics.

### Precision

{: .definition }
> **Precision** measures the proportion of positive predictions that were actually correct:
>
> $$\text{Precision} = \frac{TP}{TP + FP}$$
>
> Precision answers the question: "Of all the samples I predicted as positive, how many were actually positive?"

High precision means low false positive rate—when the model predicts positive, it's usually correct.

### Recall (Sensitivity)

{: .definition }
> **Recall** measures the proportion of actual positive samples that were correctly identified:
>
> $$\text{Recall} = \frac{TP}{TP + FN}$$
>
> Recall answers the question: "Of all the actual positive samples, how many did I correctly identify?"

High recall means low false negative rate—the model successfully finds most of the positive samples.

### F1-Score

{: .definition }
> **F1-Score** is the harmonic mean of precision and recall:
>
> $$\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

The F1-score provides a single metric that balances both precision and recall. It's particularly useful when you need to find an optimal balance between precision and recall.

## Multi-Class Classification Metrics

For multi-class problems, precision, recall, and F1-score can be computed in different ways:

### Macro-Average

Calculate the metric for each class independently and then take the unweighted average. This treats all classes equally, regardless of their size:

$$\text{Macro-Precision} = \frac{1}{c} \sum_{i=1}^{c} \text{Precision}_i$$

**Use when:** All classes are equally important, regardless of how many samples each has.

### Micro-Average

Calculate the metric globally by aggregating the contributions of all classes. This metric is dominated by the performance on the most populated classes:

$$\text{Micro-Precision} = \frac{\sum_{i=1}^{c} TP_i}{\sum_{i=1}^{c} (TP_i + FP_i)}$$

**Use when:** You want overall performance across all predictions, giving more weight to common classes.

### Weighted Average

Calculate the metric for each class and average them, weighted by the number of true instances for each class. This accounts for class imbalance:

$$\text{Weighted-Precision} = \sum_{i=1}^{c} \frac{n_i}{N} \times \text{Precision}_i$$

where $n_i$ is the number of samples in class $i$ and $N$ is the total number of samples.

**Use when:** You want a balanced metric that considers both class importance and class frequency.

## When to Use Which Metric

- **Accuracy**: Use when classes are balanced and all classes are equally important
- **Precision**: Use when false positives are costly (e.g., spam detection—marking good emails as spam is annoying)
- **Recall**: Use when false negatives are costly (e.g., medical diagnosis—missing a disease is dangerous)
- **F1-Score**: Use when you need a balance between precision and recall, especially with imbalanced datasets

## Expected Knowledge

Answer the following questions to test your understanding of classification metrics.

1. **Metric Selection:** A robot in a warehouse is tasked with identifying fragile packages.
   - **Scenario A:** If the robot misses a fragile package (a False Negative), the package is destroyed by other machinery.
   - **Scenario B:** If the robot incorrectly labels a normal package as fragile (a False Positive), the package is sent for slower, manual handling, causing a minor delay.
   
   Between **Precision** and **Recall**, which metric is more critical to maximize in this application? Justify your answer by referencing the costs of False Positives and False Negatives.

2. **Calculation and Interpretation:** For a 3-class problem, a model produces the following confusion matrix on the test set:

   |          | Predicted A | Predicted B | Predicted C |
   | :------- | :---------: | :---------: | :---------: |
   | **True A** |     80      |     15      |      5      |
   | **True B** |      8      |     70      |     12      |
   | **True C** |      2      |      5      |     93      |

   - Calculate the overall **accuracy** of the model.
   - Calculate the **Precision** and **Recall** for **Class B**.
   - Based on your calculations, is the model better at not mislabeling other classes *as Class B* (Precision), or is it better at finding all the *actual Class B* samples (Recall)?

3. **Accuracy's Pitfall:** Describe a real-world robotics scenario where a classification model could achieve 99% accuracy but still be considered a failure. What metric, such as the **F1-Score**, would better reveal the model's poor performance in this scenario, and why?