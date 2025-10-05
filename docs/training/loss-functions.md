---
title: Loss Functions
layout: default
nav_order: 1
parent: Training
---

# Loss Functions
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Machine Learning from a Statistical Perspective

In machine learning, we can adopt a statistical viewpoint where $$\boldsymbol{x}$$ and $$y$$ are random variables connected by an **unknown** joint probability distribution function $$p^{*}(\boldsymbol{x}, y)$$. We can only gather examples $$(\boldsymbol{x}, y)$$ from this distribution. Our objective is to learn a function $$\mathcal{p}(\boldsymbol{x}, y\vert \boldsymbol{w})$$ that approximates the true distribution $$p^{*}(\boldsymbol{x}, y)$$, allowing us to make predictions on new, unseen data.

{: .definition }
> **Machine Learning Task**: The goal is to approximate the real **unknown** probability distribution 
> 
> $$p^{*}(\boldsymbol{x}, y), \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{N}$$
>
> with another probability distribution
>
> $$\mathcal{p}(\boldsymbol{x}, y\vert \boldsymbol{w}), \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{N}, \quad \boldsymbol{w} \in \mathbb{R}^{m}$$
>
> parameterized by $$\boldsymbol{w}$$.

## Kullback-Leibler (KL) Divergence

To deepen our understanding, we examine the **Kullback-Leibler (KL) Divergence**.

{: .definition }
> **Kullback-Leibler (KL) Divergence**: Denoted as $$D_{\text{KL}}(p \vert\vert q)$$, it measures how one probability distribution $$p$$ differs from another probability distribution $$q$$. For discrete distributions $$p$$ and $$q$$, the formula is:
>
> $$D_{\text{KL}}(p \vert\vert q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}.$$

While this definition might not seem intuitive at first—especially since KL divergence is not symmetric, unlike many distance measures—let’s break it down with an example.

{: .note }
> The term **divergence** implies asymmetry. Unlike **metrics**, divergences are not required to be symmetric.

Consider a coin with two sides. Define a random variable $$x \in \{0, 1\}$$ (heads or tails) and assume the coin is fair, so the probability distribution $$p$$ is:

<br>

$$p(x) = \begin{cases}
    \frac{1}{2}, \quad x = 1 \\
    \frac{1}{2}, \quad x = 0
\end{cases}$$

<br>

This means the probability of heads or tails is each $$\frac{1}{2}$$. Now, consider another distribution $$q$$:

<br>

$$q(x) = \begin{cases}
    \frac{1}{3}, \quad x = 1 \\
    \frac{2}{3}, \quad x = 0
\end{cases}$$

<br>

To measure how different these distributions are, you might think to simply sum the differences between their probabilities. However, this is not typically how differences between probability distributions are measured in practice.

Instead, consider sampling data from distribution $$p$$ and calculating the likelihood of these samples under both distributions $$p$$ and $$q$$. For example, if we flip the coin ten times and observe:

<br>

$$x_{1} = 0, \quad x_{2} = 1, \quad x_{3} = 0, \quad x_{4} = 1, \quad x_{5} = 0,$$

$$x_{6} = 1, \quad x_{7} = 0, \quad x_{8} = 1, \quad x_{9} = 0, \quad x_{10} = 1$$

<br>

With five heads and five tails, representing distribution $$p$$, we can calculate the probability of these observations under $$p$$ and $$q$$:

<br>

$$p(\text{Observations} \vert p) = p(0) \cdot p(1) \cdot \ldots \cdot p(1)$$

$$p(\text{Observations} \vert q) = q(0) \cdot q(1) \cdot \ldots \cdot q(1)$$

<br>

The ratio of these probabilities is:

<br>

$$\frac{p(\text{Observations} \vert p)}{p(\text{Observations} \vert q)} = \frac{\Pi_{i=1}^{10}p(x_i)}{\Pi_{i=1}^{10}q(x_i)} = \frac{p(0)^{n_0}p(1)^{n_1}}{q(0)^{n_0}q(1)^{n_1}}$$

<br>

where $$n_0$$ is the number of tails and $$n_1$$ is the number of heads. Normalizing and taking the logarithm, we get:

<br>

$$
\begin{aligned}
&\log\biggl(\frac{p(\text{Observations} \vert p)}{p(\text{Observations} \vert q)}\biggl)^{\frac{1}{N}} = \frac{1}{N}\cdot\log\biggl(\frac{p(Observations \vert p)}{p(Observations \vert q)}\biggl) \\ \\
&=\frac{1}{N}\cdot\log\biggl(\frac{p(0)^{n_0}p(1)^{n_1}}{q(0)^{n_0}q(1)^{n_1}}\biggl) = \frac{1}{N}\log{p(0)^{n_0}} + \frac{1}{N}\log{p(1)^{n_1}} - \frac{1}{N}\log{q(0)^{n_0}} - \frac{1}{N}\log{q(1)^{n_1}} \\ \\
&=\frac{n_0}{N}\log{p(0)} + \frac{n_1}{N}\log{p(1)} - \frac{n_0}{N}\log{q(0)} - \frac{n_1}{N}\log{q(1)} \\ \\
&= \frac{n_0}{N}\log\frac{p(0)}{q(0)} + \frac{n_1}{N}\log\frac{p(1)}{q(1)}
\end{aligned}
$$

<br>

As $$N$$ approaches infinity, $$\frac{n_0}{N}$$ converges to $$p(0)$$ and $$\frac{n_1}{N}$$ converges to $$p(1)$$, leading to:

<br>

$$ \frac{n_0}{N}\log\frac{p(0)}{q(0)} + \frac{n_1}{N}\log\frac{p(1)}{q(1)} \quad \xrightarrow{N \to \infty} \quad p(0) \cdot \log\frac{p(0)}{q(0)} + p(1) \cdot \log\frac{p(1)}{q(1)}$$

<br>

This is precisely **Kullback-Leibler (KL) Divergence**!

{: .slogan }
> The **KL Divergence** between probability distributions $$p$$ and $$q$$:
>
> $$D_{\text{KL}}(p \vert\vert q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}$$
>
> quantifies how "**surprised**" we would be if samples from distribution $$p$$ were claimed to come from distribution $$q$$. The smaller the **KL Divergence**, the more similar the distributions are.
>
> {: .note }
> The term [Surprisal](https://en.wikipedia.org/wiki/Information_content) refers to the amount of information gained, commonly known as **Information content** or **Shannon information**.

Understanding the **KL Divergence** helps us see why it is used in machine learning: we sample data from the distribution $$p^{*}(\boldsymbol{x}, y)$$ and measure how **surprised** we would be if these samples came from the distribution $$p(\boldsymbol{x}, y \vert \boldsymbol{w})$$ that we have learned.

## Cross-Entropy Loss

Let's consider our probability distributions within the context of **KL Divergence** $$D_{\text{KL}}(p^* \,\vert\vert \,p)$$.

<br>

$$
\begin{aligned}
&D_{\text{KL}}(p^* \,\vert\vert \,p) = \sum_{(\boldsymbol{x}, y) \sim p^{*}} p^*(y, \boldsymbol{x}) \cdot \log \biggl(\frac{p^*(y, \boldsymbol{x})}{p(y, \boldsymbol{x} \,\vert\, \boldsymbol{w})}\biggr) = \\ \\
&= \sum_{(\boldsymbol{x}, y) \sim p^{*}} p^*(y, \boldsymbol{x}) \cdot \biggl[\log p^*(y, \boldsymbol{x}) - \log\Bigl(p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \cdot p(\boldsymbol{x})\Bigr)\biggr] = \\ \\
&= \underbrace{\sum_{(\boldsymbol{x}, y)} p^*(y, \boldsymbol{x}) \cdot \log p^*(y, \boldsymbol{x})}_{\text{Constant term}} - \sum_{(\boldsymbol{x}, y)} p^*(y, \boldsymbol{x}) \cdot \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) - \underbrace{\sum_{(\boldsymbol{x}, y)} p^*(y, \boldsymbol{x}) \cdot \log p(\boldsymbol{x})}_{\text{Constant term}} = \\ \\
&= - \sum_{(\boldsymbol{x}, y) \sim p^{*}} p^*(y, \boldsymbol{x}) \cdot \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w})
\end{aligned}
$$

<br>

Thus, minimizing the **Cross-Entropy Loss** $$H(p^*, p)$$ is equivalent to minimizing the **KL Divergence** $$D_{\text{KL}}(p^* \,\vert\vert \,p)$$. This is why we use the **Cross-Entropy Loss** in practice.

<br>

$$\arg\min_{\boldsymbol{w}} D_{\text{KL}}(p \,\vert\vert \,p^*) = \arg\min_{\boldsymbol{w}} - \sum_{(\boldsymbol{x}, y)} p^*(y, \boldsymbol{x}) \cdot \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) = \arg\min_{\boldsymbol{w}} H(p^*, p)$$

<br>

Directly minimizing the **Cross-Entropy** is challenging because the true distribution $$p^*(y, \boldsymbol{x})$$ is unknown. We can approximate it as follows:

<br>

$$
\begin{aligned}
& \arg\min_{\boldsymbol{w}} \Bigl( - \sum_{(\boldsymbol{x}, y) \sim p^{*}} p^*(y, \boldsymbol{x}) \cdot \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \Bigr) = \arg\min_{\boldsymbol{w}} \Bigl( - \mathbb{E}_{(\boldsymbol{x}, y) \sim p^{*}} [\log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w})] \Bigr) \approx \\ \\
&\approx \arg\min_{\boldsymbol{w}} \Bigl( \frac{1}{N} \sum_{(\boldsymbol{x}, y) \sim \mathcal{D}} -\log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \Bigr) = \arg\min_{\boldsymbol{w}} \Bigl( - \sum_{(\boldsymbol{x}, y) \sim \mathcal{D}} \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \Bigr)
\end{aligned}
$$

<br>

where $$\mathcal{D}$$ is the dataset and $$N$$ is the number of samples in the dataset.

This step is where theory meets practice. Since we cannot compute the true expectation over the unknown distribution $$p^*$$, we approximate it with an average over the dataset $$\mathcal{D}$$, which is a sample from $$p^*$$. This is a form of **Monte Carlo estimation**.

{: .definition }

>The **Cross-Entropy Loss** from observed data from the true distribution $$p^*(y \,\vert\, \boldsymbol{x})$$ is defined as
>
> $$- \frac{1}{N} \sum_{(\boldsymbol{x}, y) \sim \mathcal{D}} \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w})$$
>
> where $$\mathcal{D}$$ is the dataset and $$N$$ is the number of samples in the dataset.

### Maximum Likelihood Estimation (MLE)

The **Cross-Entropy Loss** is closely related to **Maximum Likelihood Estimation (MLE)**. When we use MLE, we aim to find the parameters $$\boldsymbol{w}$$ that maximize the likelihood of the observed data. This is equivalent to minimizing the negative log-likelihood of the data, which leads to the same objective as minimizing the Cross-Entropy Loss. 

<br>

$$
\begin{aligned}
&\arg\min_{\boldsymbol{w}} \Bigl( - \sum_{(\boldsymbol{x}, y) \sim \mathcal{D}} \log p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \Bigr) = \arg\min_{\boldsymbol{w}} \Bigl( -\log \prod_{(\boldsymbol{x}, y) \sim \mathcal{D}} p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \Bigr) =  \\ \\
&= \arg\max_{\boldsymbol{w}} \Bigl( \prod_{(\boldsymbol{x}, y) \sim \mathcal{D}} p(y \,\vert\, \boldsymbol{x}, \boldsymbol{w}) \Bigr)
\end{aligned}
$$

<br>

## Softmax Function

As previously mentioned, the output of a classifier is a vector of unnormalized scores, or logits, $$\boldsymbol{s}$$. To convert these logits into probabilities, we use the softmax function $$\boldsymbol{\sigma}(\boldsymbol{s})$$.

<br>
<div align="center">
    <img src="{{ site.baseurl }}/assets/images/statistical_model.png" width="600px"/>
</div>
<br>

{: .definition }
> The **Softmax Function**, denoted as $$\boldsymbol{\sigma}(\boldsymbol{s})$$, is defined as:
> 
>$$\boldsymbol{\sigma}(\boldsymbol{s})_i = \frac{e^{s_i}}{\sum_{j=1}^{c} e^{s_j}}$$
>
> where $$c$$ is the length of the vector $$\boldsymbol{s}$$.

The softmax function takes the logits $$\boldsymbol{s}$$ as input and outputs a vector of probabilities that sum to one. This function is used to convert the raw scores output by the classifier into probabilities, representing the model's confidence in each class and used to make predictions.


## Expected Knowledge

Answer the following questions to test your understanding of the theoretical basis for loss functions.

1. **KL Divergence vs. Cross-Entropy:** In training, our goal is to make our model's distribution $$p$$ as close as possible to the true data distribution $$p^*$$. While KL Divergence ($$D_{\text{KL}}(p^* || p)$$) directly measures this, in practice we minimize the Cross-Entropy Loss ($$H(p^*, p)$$). Based on the derivation, explain *why* minimizing cross-entropy is equivalent to minimizing KL divergence. What term from the KL divergence formula can we ignore during optimization and why?

2. **The Role of Softmax:** The cross-entropy loss function requires a probability distribution as input from our model. What specific function do we use to convert the raw `logits` from a neural network into a valid probability distribution? Describe one key property of this function's output.

3. **Intuition and Application:** What is the relationship between minimizing cross-entropy loss and the principle of Maximum Likelihood Estimation (MLE)? Explain it conceptually. If you were building a classifier from scratch, what principle does this connection justify about your choice of loss function?

4. **KL Divergence Asymmetry:** KL Divergence is not a true "distance" metric because it is asymmetric, i.e., $$D_{\text{KL}}(p || q) \neq D_{\text{KL}}(q || p)$$. What does the value $$D_{\text{KL}}(p || q)$$ intuitively represent, in terms of information or "surprise"?