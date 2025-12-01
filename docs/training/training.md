---
title: Training
layout: default
has_children: true
nav_order: 4
mathjax: true
---

# Model Training
{: .no_toc }

This section covers how machine learning models learn from data, including the mathematical foundations and algorithms that make learning possible.

## What You'll Learn

- **[Loss Functions]({{ site.baseurl }}{% link docs/training/loss-functions.md %})** - How we measure model performance and define learning objectives
- **[Gradient Descent]({{ site.baseurl }}{% link docs/training/gradient-descent.md %})** - The fundamental optimization algorithm for training models
- **[Optimization]({{ site.baseurl }}{% link docs/training/optimization.md %})** - Convergence rate, oscillations, and diminishing gradients in optimization
- **[Backpropagation]({{ site.baseurl }}{% link docs/training/backpropagation.md %})** - How gradients are computed efficiently in neural networks
- **[Data Augmentation]({{ site.baseurl }}{% link docs/training/data_augmentation.md %})** - Techniques to artificially expand training datasets
- **[Metric Learning]({{ site.baseurl }}{% link docs/training/metric_learning.md %})** - Learning similarity measures for better representations
- **[Neural Network Training Fundamentals]({{ site.baseurl }}{% link docs/training/net_training_fundamentals.md %})** - Core principles of training neural networks

Understanding these concepts is crucial for training any machine learning model effectively. When a model fails to learn, the problem often lies not in the model's architecture, but in the training process itself. Mastering these fundamentals will give you the tools to diagnose and solve these issues.