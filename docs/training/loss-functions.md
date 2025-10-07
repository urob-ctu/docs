---
title: Loss Functions
layout: default
nav_order: 1
parent: Training
---

# Loss Functions

{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

### An Example: A Robot Identifying Tools
>
> Before we dive into the math, let's use a simple robotics scenario. Imagine a robot on an assembly line that needs to identify tools placed in front of its camera. It must distinguish between a 'wrench', a 'screwdriver', and a 'hammer'.
>
> In this machine learning problem:
>
> * `x` is the **input data**: This is the image from the robot's camera.
> * `y` is the **output label**: This is the name of the tool, e.g., 'wrench'.
> * `w` are the **model's parameters**: These are all the tunable "knobs" of our machine learning model. If we are using a neural network, `w` represents all of its weights and biases. The values of `w` determine the model's behavior.
>
> Our goal is to create a model that, given an image `x`, predicts the correct label `y`. Because the world is complex and measurements are noisy, we frame this in terms of probability. We want our model to calculate:
>
> **`p(y | x, w)`**
>
> This is a **conditional probability**. It reads as: "The probability of the tool being `y` (e.g., 'wrench'), **given** the input image `x` and our model's current parameters `w`."
>
> For a single image `x`, our model will output three probabilities:
>
> * `p(y='wrench' | x, w)` = 0.7
> * `p(y='screwdriver' | x, w)` = 0.2
> * `p(y='hammer' | x, w)` = 0.1
>
> The goal of training is to adjust the parameters `w` so that for every image of a wrench we see, the model assigns the highest possible probability to the correct label.

---

### Machine Learning via Maximum Likelihood

How do we choose the best settings (parameters, `w`) for our model?
The usual answer in machine learning is simple: pick the settings that make the data we actually saw **most likely**.
This principle is called **Maximum Likelihood Estimation (MLE)**.

Formally, if our data samples $$(\boldsymbol{x}_i, y_i)$$ are independent, the total likelihood of our model parameters is the product of the probabilities our model assigns to each sample:

$$ \text{Likelihood}(\boldsymbol{w}) = p_{\text{model}}(\mathcal{D} | \boldsymbol{w}) = \prod_{i=1}^{N} p(y_i | \boldsymbol{x}_i, \boldsymbol{w}) $$

Our goal is to find the parameters that make this likelihood as large as possible:

$$ \boldsymbol{w}_{\text{MLE}} = \arg\max_{\boldsymbol{w}} \prod_{i=1}^{N} p(y_i | \boldsymbol{x}_i, \boldsymbol{w}) $$

---

### The Log-Likelihood Trick

Multiplying thousands or millions of small probability values is difficult numerically and mathematically. A long product can easily result in a number so small it can't be stored accurately (arithmetic underflow).

To fix this, we work with the **logarithm** of the likelihood instead.

Why does this help?

* **Logs turn multiplication into addition**, which is computationally stable and much easier to differentiate.
* **The logarithm is a strictly increasing function.** This means that if `likelihood_A > likelihood_B`, then `log(likelihood_A) > log(likelihood_B)`. Maximizing the log-likelihood is the same as maximizing the likelihood.

So instead of maximizing a product, we maximize a sum: the **Log-Likelihood**.

$$ \mathcal{LL}(\boldsymbol{w}) = \sum_{i=1}^{N} \log p(y_i | \boldsymbol{x}_i, \boldsymbol{w}) $$

---

### Connecting to Loss Functions

So far, we’ve been talking about maximizing the log-likelihood. But in practice, training algorithms like gradient descent are designed to **minimize** a *loss function*.

That’s an easy fix—we just flip the sign and minimize the **Negative Log-Likelihood (NLL)**:

$$ -\sum_{i=1}^{N} \log p(y_i | \boldsymbol{x}_i, \boldsymbol{w}) $$

There’s one final adjustment: we usually **divide by the number of samples, $$N$$**.
This gives us the *average* loss per sample, which allows us to compare model performance across datasets of different sizes. A total loss of 500 is bad for 100 samples but great for a million; an average loss is a more consistent metric.

This gives us our final loss function, the **Negative Log-Likelihood Loss**:

$$ \mathcal{L}_{\text{NLL}} = -\frac{1}{N} \sum_{i=1}^{N} \log p(y_i | \boldsymbol{x}_i, \boldsymbol{w}) $$

This is the loss function used by most classifiers, famously known as **Cross-Entropy Loss**.

### The Theoretical Justification: Kullback-Leibler (KL) Divergence

We've established a practical procedure (MLE) that leads to our cross-entropy loss function. But *why* is maximizing the likelihood the correct goal from a statistical point of view? The answer lies in **Kullback-Leibler (KL) Divergence**, which provides the theoretical foundation for this approach.

Let's build the idea of KL Divergence from the ground up by first considering a simple case.

Imagine we have a "true" probability distribution, let's call it `p_data`, which generates outcomes. We also have our model, `p`, which tries to approximate it. We want to measure how similar our model `p` is to the true distribution `p_data`.

Suppose we draw a long sequence of `N` samples from the true distribution: `A = (y_1, y_2, ..., y_N)`. A good model `p` should assign a high probability to this sequence, ideally one that is close to the true probability assigned by `p_data`.

A way to measure the difference in their perspectives is to look at the **difference between their log-likelihoods** for the sequence. Our goal is to make this difference as small as possible. Using the logarithm rule $$\log(a) - \log(b) = \log(a/b)$$, this difference becomes:

$$ \sum_{k=1}^{N} \log p_{\text{data}}(y_k) - \sum_{k=1}^{N} \log p(y_k) = \sum_{k=1}^{N} \log\left(\frac{p_{\text{data}}(y_k)}{p(y_k)}\right) $$

This sum depends on the length of our sample sequence, `N`. To create a stable metric that doesn't depend on the dataset size, we must normalize by `N` to get the **average log-probability ratio**:

$$ \frac{1}{N} \sum_{k=1}^{N} \log\left(\frac{p_{\text{data}}(y_k)}{p(y_k)}\right) $$

Now for the crucial step. The Law of Large Numbers states that as we collect an infinite amount of data ($$N \to \infty$$), this sample average converges to its theoretical expectation, taken over the true distribution `p_data`.

$$ \mathbb{E}_{y \sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(y)}{p(y)}\right] = \sum_{y} p_{\text{data}}(y) \log\left(\frac{p_{\text{data}}(y)}{p(y)}\right) $$

This final expression is the general formula for **Kullback-Leibler (KL) Divergence**. It represents the "extra surprise" or information loss when we use distribution `p` to model `p_data`.

**Now, let's apply this concept to our actual robotics problem.** In our case, the probabilities are not simple; they are **conditional** on the input `x` (the camera image).
For any given input `x`, we are comparing:

* `p_data(y | x)`: The **true, ideal distribution** of labels `y` for that input `x`.
* `p(y | x, w)`: Our **model's approximation** for that `x`.

So, we take the general KL Divergence formula and substitute these conditional distributions into it. This gives us the divergence *for a specific input `x`*.

{: .definition }
> **Kullback-Leibler (KL) Divergence**: Denoted as $$D_{\text{KL}}(p_{\text{data}} \parallel p)$$, it measures how a probability distribution `p` diverges from a reference probability distribution `p_data`. For our machine learning context with a given input `x`, the formula is:
>
>$$
D_{\text{KL}}\left(p_{\text{data}}(y|x) \parallel p(y|x, w)\right) = \sum_{y \in \mathcal{Y}} p_{\text{data}}(y|x) \log \left(\frac{p_{\text{data}}(y|x)}{p(y|x, w)}\right)
>$$

Our overall goal during training is to find the parameters `w` that minimize this divergence on average over all the possible inputs `x` we might encounter.

### The Big Connection

Minimizing the KL Divergence is the correct theoretical goal. But how does this connect back to the loss function we use in practice?

Let's expand the KL Divergence formula for a single input `x`:

$$
D_{\text{KL}}(p_{\text{data}}(y|x) \parallel p(y|x,w)) = \sum_{y} p_{\text{data}}(y|x) \log p_{\text{data}}(y|x) - \sum_{y} p_{\text{data}}(y|x) \log p(y|x,w)
$$

Look closely at the two parts:

1. `Σ p_data(y|x) log p_data(y|x)`: This is the **entropy** of the true data distribution for a given `x`. From our model's perspective, this is a fixed value. We cannot change it by adjusting our model's parameters `w`.
2. `-Σ p_data(y|x) log p(y|x,w)`: This is the **cross-entropy**. It depends on our model's predictions `p(y|x,w)` and is the only part we can minimize.

So, minimizing KL Divergence is equivalent to minimizing the cross-entropy. But we don't know `p_data(y|x)`.

This is where our training data $$(\boldsymbol{x}_i, y_i)$$ comes in. For a training sample, we assume the ground truth label $$y_i$$ is the *only* correct outcome. We can represent this "true" distribution `p_data(y | x_i)` as a **one-hot encoded vector**: it's 1 for the true class $$y_i$$ and 0 for every other class.

Let's see what happens to the cross-entropy sum with this one-hot assumption:

$$
-\sum p_{\text{data}}(y \vert x_i) log p(y \vert x_i,w)
$$

In this sum, the term `p_data(y|x_i)` is 0 for all `y` that are not the true label $$y_i$$. This makes all parts of the sum zero, *except* for the one where `y` = $$y_i$$. For that single term, `p_data(y_i|x_i)` is 1. The entire sum collapses to:

$$-1 \cdot \log p(y_i | \boldsymbol{x}_i, \boldsymbol{w})$$

This is simply the negative log-probability of the correct class for a single sample! To get our final loss for the entire dataset, we just average this over all `N` samples:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p(y_i | \boldsymbol{x}_i, \boldsymbol{w})
$$

This is exactly the **Negative Log-Likelihood (NLL)**, or Cross-Entropy loss. This provides the beautiful theoretical link: minimizing the NLL on our training data is a practical way of minimizing the KL Divergence between our model's distribution and the true underlying data distribution.
