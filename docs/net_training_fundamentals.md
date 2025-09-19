---
title: Neural Network Training Fundamentals
layout: default
has_children: false
nav_order: 6
mathjax: true
---

# Neural Network Training Fundamentals

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

Understanding the fundamentals of neural network training is crucial for building effective models. This page covers essential concepts including loss initialization, weight initialization, activation functions, and training stability techniques that apply to any neural network architecture.

## Loss Initialization

{: .definition }
>**Loss initialization** refers to the expected loss value when a model makes random predictions at the start of training. Understanding this baseline helps detect initialization problems and set reasonable training expectations.

For a classification problem with $N$ classes, if the model outputs uniform random predictions, the expected loss is:

$$ \ell_{expected} = -\log\left(\frac{1}{N}\right) = \log(N) $$

In our case, a 30-class fruit classification task: $\ell_{expected} \approx \log(30) \approx 3.40$

### Random initialization Issues

{: .important }
>**Extreme logit values** lead to:
>- **Overconfident wrong predictions** on random data
>- **High initial loss** much greater than $\log(N)$
>- **Slow convergence** due to gradient saturation

## Weight Initialization

Proper weight initialization is crucial for stable training and convergence. 

{: .important }
>When multiplying many random matrices (layers), the variance grows or shrinks out of control, leading to unstable activations and gradients.

**The Mathematical Issue:**
```
input ~ N(0, 1)        # Standard normal
weights ~ N(0, 1)      # Standard normal  
output = input @ weights

Result: Var(output) = n × Var(input) × Var(weights) = n
```

When we have 10 inputs: output variance becomes 10× larger!
When we have 12,288 inputs: output variance becomes 12,288× larger! 



### Xavier/Glorot Initialization

{: .definition }
>**Xavier (Glorot) initialization** scales weights to maintain unit variance through layers, preventing activation values from growing or shrinking exponentially.

#### Historical Evolution

**LeCun Normal (1998)** - First systematic approach:
$$ W \sim \mathrm{N}\left(0, \frac{1}{n_{in}}\right) $$

**Glorot/Xavier (2010)** - Improved version considering both dimensions:
$$ W \sim \mathrm{N}\left(0, \frac{2}{n_{in} + n_{out}}\right) $$

{: .important }
>**Key Insight:** This ensures that the variance of activations remains approximately constant across layers, preventing the **exploding/vanishing gradient problem**.

### Kaiming / He Initialization
Kaiming initialization (He et al., 2015) was developed for networks with ReLU and ReLU-like activations. These nonlinearities discard or suppress part of the input (e.g. ReLU sets negatives to zero), which reduces the variance of activations. The initialization adjusts for this effect.

#### Activation-Specific Formulas

**For ReLU (and ReLU-like) networks:**
$$
W \sim \mathrm{N}\!\left(0, \frac{2}{n_{in}}\right)
$$

**For Tanh networks (scaled Xavier variant):**
$$
W \sim \mathrm{N}\!\left(0, \frac{(5/3)^2}{n_{in}}\right)
$$

**Why the factor of 2?**  
With ReLU, roughly half of the inputs are zeroed out, cutting the variance in half. The factor of 2 counterbalances this so that the output variance remains stable.

## Activation Functions

### Tanh Activation

{: .definition }
>The **hyperbolic tangent (tanh)** activation function maps inputs to the range $(-1, 1)$ and provides zero-centered outputs.

$$ \text{tanh}(x) = \frac{e^{2x} - 1}{e^{2x} + 1} $$

**Derivative:**
$$ \frac{d}{dx}\text{tanh}(x) = 1 - \text{tanh}^2(x) $$

### Activation Saturation

{: .important }
>**Activation saturation** occurs when neurons output values at the extremes of the activation function's range, causing gradients to approach zero and learning to slow dramatically.

For tanh activation, saturation happens when:

- **Large inputs**: When `|x|` is large → `tanh(x) ≈ ±1` (saturated)
- **Zero gradients**: When saturated → `tanh'(x) ≈ 0` (no gradient flow)

This leads to the **vanishing gradient problem** where information doesn't propagate backward effectively.

## Batch Normalization

{: .definition }
>**Batch Normalization** is a technique that normalizes layer inputs by adjusting and scaling activations. It stabilizes training by reducing internal covariate shift.

### Mathematical Formulation

For a batch of activations $x$:

**Step 1: Normalize**
$$ x' = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$

Where:
- $\mu_B$ = batch mean
- $\sigma_B^2$ = batch variance  
- $\epsilon$ = small constant for numerical stability

**Step 2: Scale and Shift**
$$ y = \gamma x' + \beta $$

Where:
- $\gamma$ = learnable scale parameter (initially 1)
- $\beta$ = learnable shift parameter (initially 0)

### Benefits of Batch Normalization

1. **Stabilizes training** by normalizing layer inputs
2. **Enables higher learning rates** due to improved gradient flow
3. **Reduces sensitivity** to weight initialization  
4. **Acts as regularization** by adding noise through batch statistics
5. **Accelerates convergence** in most cases

### Inference Mode

{: .important }
>During inference, batch statistics may not be available (single sample prediction). We need to use fixed statistics instead of batch-dependent ones.

**Solution 1: Compute from Whole Training Set**

After training is complete, compute statistics on the entire training dataset:

```python
# After training - compute statistics from all training data
with torch.no_grad():
    all_activations = []
    for batch in training_loader:
        activations = model.get_layer_activations(batch)
        all_activations.append(activations)
    
    # Compute global training statistics
    training_mean = torch.cat(all_activations, dim=0).mean(dim=0)
    training_std = torch.cat(all_activations, dim=0).std(dim=0)
```

{: .important }
>**Drawback:** This approach is slow and requires processing the entire training set after training is complete.

**Solution 2: Running Averages (Preferred)**

Maintain exponentially weighted moving averages during training:

$$ \mu_{running} = \alpha \mu_{running} + (1-\alpha) \mu_{batch} $$
$$ \sigma_{running} = \alpha \sigma_{running} + (1-\alpha) \sigma_{batch} $$

Where $\alpha$ is the momentum parameter (typically 0.9).

### Removing Bias from Linear Layers

{: .important }
>When using batch normalization, the bias term from the preceding linear layer becomes redundant and can be removed for efficiency.

**Mathematical reasoning:**
```python
# Standard approach:
linear_output = X @ W + b          # Linear layer with bias
normalized = (linear_output - μ) / σ  # Batch norm removes the mean anyway!
final = normalized * γ + β           # β acts as the new bias
```

**Optimized approach:**
```python
# Remove bias from linear layer
linear_output = X @ W               # No bias needed!
normalized = (linear_output - μ) / σ
final = normalized * γ + β          # β provides the bias functionality
```

**Key insight:** Since batch normalization subtracts the mean, any constant bias gets removed anyway. The learnable β parameter in batch norm serves as the bias term.

## Training Considerations

### Learning Rate Scheduling

{: .definition }
>**Learning rate scheduling** involves adjusting the learning rate during training to improve convergence and final performance.

Common strategies:
- **Step decay**: Reduce learning rate by factor at fixed intervals
- **Exponential decay**: Gradually decrease learning rate exponentially
- **Cosine annealing**: Follow cosine curve for smooth transitions



## Expected Knowledge

- **Initialization Theory** - Understanding Xavier/Glorot and Kaiming initialization methods
- **Activation Functions** - Properties of tanh, ReLU, and saturation effects  
- **Batch Normalization** - Mathematical formulation and benefits for training stability
- **Forward/Backward Pass** - Step-by-step computation through MLP layers
- **Gradient Descent** - Manual parameter updates and learning rate effects
- **Data Augmentation** - Transformation techniques and proper application methodology
- **Training Dynamics** - Loss initialization, convergence patterns, and debugging techniques
