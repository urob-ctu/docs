---
title: Regression
layout: default
nav_order: 3
parent: Fundamentals
mathjax: true
---

# Regression
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

While classification predicts discrete class labels, regression is another fundamental supervised learning task that predicts continuous numerical values. For example, predicting house prices, temperature, stock prices, or the trajectory of a robot arm.

## Definition of Regression

{: .definition }
>**Regression** is the task of mapping from a feature space to a continuous output space. The goal is to learn a function $\mathcal{F}$ that maps features to continuous values.
>
>$$\mathcal{F}: \mathcal{X} \rightarrow \mathbb{R}$$
>
>$$\mathcal{F}(\boldsymbol{x}) = y, \quad \boldsymbol{x} \in \mathcal{X}, \quad y \in \mathbb{R}$$

For this course, we assume:
- The feature space is a vector space: $\mathcal{X} = \mathbb{R}^{d}$
- The output is a real number: $y \in \mathbb{R}$ (or vector $\boldsymbol{y} \in \mathbb{R}^{m}$ for multi-output regression)
- We denote predictions as $\hat{y}$ to distinguish from true values $y$

## Types of Regression

### Linear Regression

The simplest form of regression assumes a linear relationship between features and output:

$$\hat{y} = \boldsymbol{w}^T\boldsymbol{x} + b = \sum_{i=1}^{d} w_i x_i + b$$

where:
- $\boldsymbol{w} \in \mathbb{R}^{d}$ is the weight vector
- $b \in \mathbb{R}$ is the bias (intercept)
- $\boldsymbol{x} \in \mathbb{R}^{d}$ is the feature vector

### Polynomial Regression

Extends linear regression by transforming features into polynomial terms:

$$\hat{y} = w_0 + w_1x + w_2x^2 + ... + w_nx^n$$

This is still linear in parameters but can model non-linear relationships.

### Non-linear Regression

Uses non-linear functions like neural networks to model complex relationships:

$$\hat{y} = f(\boldsymbol{x}; \boldsymbol{\theta})$$

where $f$ is a non-linear function parameterized by $\boldsymbol{\theta}$.

## Loss Functions for Regression

Unlike classification which uses cross-entropy loss, regression typically uses different loss functions:

### Mean Squared Error (MSE)

The most common loss function for regression:

{: .definition }
>**Mean Squared Error** measures the average squared difference between predictions and true values:
>
>$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

MSE heavily penalizes large errors due to the squaring operation.

### Mean Absolute Error (MAE)

{: .definition }
>**Mean Absolute Error** measures the average absolute difference:
>
>$$\mathcal{L}_{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

MAE is more robust to outliers than MSE but less smooth for optimization.

### Huber Loss

Combines the benefits of MSE and MAE:

$$\mathcal{L}_{Huber} = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

## Evaluation Metrics for Regression

### R² Score (Coefficient of Determination)

Measures the proportion of variance in the target variable explained by the model:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

where $\bar{y}$ is the mean of true values. $R^2 = 1$ indicates perfect prediction.

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

RMSE has the same units as the target variable, making it interpretable.

## Regression vs Classification

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Output** | Discrete classes | Continuous values |
| **Example** | Is this email spam? | What will the temperature be? |
| **Loss Function** | Cross-entropy | MSE, MAE |
| **Evaluation** | Accuracy, F1-score | RMSE, R² score |
| **Output Layer** | Softmax | Linear activation |

## Connection to Robotics

Regression is particularly important in robotics for:
- **Inverse Kinematics**: Predicting joint angles from desired end-effector positions
- **Force Control**: Estimating required forces for manipulation tasks
- **Trajectory Prediction**: Forecasting object or robot paths
- **Sensor Calibration**: Mapping sensor readings to physical quantities

## Expected Knowledge

Answer the following questions to test your understanding of regression.

1. **Task Identification:** You are given two robotics challenges:
   - **Task A:** Predict the exact joint angles (in degrees) required for a robotic arm to reach a target coordinate `(x, y, z)`.
   - **Task B:** Predict whether a grasped object is `stable` or `unstable` based on force sensor readings.
   
   Which task is regression and which is classification? For the regression task, what loss function (MSE or MAE) might you prefer if your sensor occasionally produces large, erroneous outlier readings, and why?

2. **Interpreting Metrics:** Your team develops a regression model to predict the remaining battery life of a drone in minutes. The model achieves an **RMSE** of 5.0 and an **R² Score** of 0.95.
   - What does an RMSE of 5.0 mean in a practical sense?
   - What does the R² score of 0.95 tell you about the model's predictive power?
   - Why are both metrics useful? What does each one tell you that the other doesn't?

3. **Model Complexity:** What is the fundamental difference between linear regression and polynomial regression? If you fit a linear regression model to data that has a clear curve, what will the model's predictions look like relative to the true data?