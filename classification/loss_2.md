
# Loss
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Machine Learning from Statistical Point of View

We can look at the machine learning task from the statistical point of view. That would mean that $$\boldsymbol{x}$$ and $$y$$ are random variables. These variales are related by an <u>unknown</u> joint probability distribution function $$p_{data}(\boldsymbol{x}, y)$$. We can only collect examples $$(\boldsymbol{x}, y)$$ from $$p_{data}(\boldsymbol{x}, y)$$. Our goal is to learn a function $$\mathcal{p}(\boldsymbol{x}, y\vert \boldsymbol{w})$$ that approximates the true distribution $$p_{data}(\boldsymbol{x}, y)$$. We can then use this function to make predictions on new, unseen data.

{: .definition }
> **Machine Learning Task**  is a task where the goal is to approximate the real <u>unknown</u> probability distribution 
> 
> $$p^{*}(\boldsymbol{x}, y), \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{N}$$
>
> by another probability distribution
>
> $$\mathcal{p}(\boldsymbol{x}, y\vert \boldsymbol{w}), \quad \boldsymbol{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{N}, \quad \boldsymbol{w} \in \mathbb{R}^{m}$$
>
> parametrized by $$\boldsymbol{w}$$.

## Kullback-Leibler (KL) Divergence

To gain a deeper knowledge, we will look at the **Kullback-Leibler (KL) Divergence**.

{: .definition }
> **Kullback-Leibler (KL) Divergence** denoted $$ D_{\text{KL}}(p \vert\vert q)$$ is a statistical distance: a measure how one probability distribution $$p$$ is different from another probability distribution $$q$$. For discrete distributions $$p$$ and $$q$$ the formula is
>
> $$D_{\text{KL}}(p \vert\vert q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}.$$

Personally I don't think this definition is super friendly. For example when I read distance, I think of a symmetric function, but KL is not symmetric. But let's try to understand this intuitively. Forget the definition for now and focus on the following example.

{: .note }
> The **divergence** means that it is not required to be symmetric. There are bunch of other statistical distances - **metrics** that are required to be symmetric.

We have a coin with two sides. Let's define a random variable $$x \in \{0, 1\}$$ (heads or tails) and let's assume that the coin is fair and so the probability distribution $$p$$ is defined as follows:

$$p(x) = \begin{cases}
    \frac{1}{2}, \quad x = 1 \\
    \frac{1}{2}, \quad x = 0
\end{cases}$$

The probability that the coin lands on heads is $$\frac{1}{2}$$ and the probability that the coin lands on tails is $$\frac{1}{2}$$. We have another probability distribution $$q$$ that is defined as follows:

$$q(x) = \begin{cases}
    \frac{1}{3}, \quad x = 1 \\
    \frac{2}{3}, \quad x = 0
\end{cases}$$

and we want to measure, how different they are. First think that comes into my mind is why don't we just sum the differences between the probability distributions?

$$\text{My Distance} = \lvert p(x=1) - q(x=1) \rvert + \lvert p(x=0) - q(x=0) \rvert = \frac{1}{2} - \frac{1}{3} + \frac{1}{2} - \frac{2}{3} = \frac{1}{6}$$

I think that this is some statistical metric, but no one really tells you why we use some KL Divergence instead of this. So now trust me for another minute and we will come to this later. What if we would sample some data from the distribution $$p$$ and then calculate the probability that the samples are from the distribution $$q$$? Let's try that. We will flip the coin ten times

$$x_{1} = 0, \quad x_{2} = 1, \quad x_{3} = 0, \quad x_{4} = 1, \quad x_{5} = 0,$$

$$x_{6} = 1, \quad x_{7} = 0, \quad x_{8} = 1, \quad x_{9} = 0, \quad x_{10} = 1$$

We can see that we have got 5 heads and 5 tails, which represents the distribution $$p$$ perfectly. Now we can compute the probability that it is sampled from the distribution $$p$$ and $$q$$.

$$p(\text{Observations} \vert p) = p(0) \cdot p(1) \cdot p(0) \cdot p(1) \cdot p(0) \cdot p(1) \cdot p(0) \cdot p(1) \cdot p(0) \cdot p(1)$$

$$p(\text{Observations} \vert q) = q(0) \cdot q(1) \cdot q(0) \cdot q(1) \cdot q(0) \cdot q(1) \cdot q(0) \cdot q(1) \cdot q(0) \cdot q(1)$$

Now we can calculate the ratio of these probabilities

$$\frac{p(\text{Observations} \vert p)}{p(\text{Observations} \vert q)} = \frac{\Pi_{i=1}^{10}p(x_i)}{\Pi_{i=1}^{10}q(x_i)} = \frac{P(0)^{n_0}p(1)^{n_1}}{q(0)^{n_0}q(1)^{n_1}}$$

where $$n_0$$ is the number of tails and $$n_1$$ is the number of heads. And we are almost there. Now this is unnormalized value, so we can normalize it by raising it to the power of samples $$1/N = 1/(n_0 + n_1)$$. Then we can also compute the logarithm of this value, which will not change the monotonicity of the function.

$$
\begin{aligned}
&\log\biggl(\frac{p(\text{Observations} \vert p)}{p(\text{Observations} \vert q)}\biggl)^{\frac{1}{N}} = \\ \\
&=\frac{1}{N}\cdot\log\biggl(\frac{p(\text{Observations} \vert p)}{p(\text{Observations} \vert q)}\biggl) = \frac{1}{N}\cdot\log\biggl(\frac{p(0)^{n_0}p(1)^{n_1}}{q(0)^{n_0}q(1)^{n_1}}\biggl) = \\ \\
&=\frac{1}{N}\log{p(0)^{n_0}} + \frac{1}{N}\log{p(1)^{n_1}} - \frac{1}{N}\log{q(0)^{n_0}} - \frac{1}{N}\log{q(1)^{n_1}} = \\ \\
&=\frac{n_0}{N}\log{p(0)} + \frac{n_1}{N}\log{p(1)} - \frac{n_0}{N}\log{q(0)} - \frac{n_1}{N}\log{q(1)} = \\ \\
&= \frac{n_0}{N}\log\frac{p(0)}{q(0)} + \frac{n_1}{N}\log\frac{p(1)}{q(1)}
\end{aligned}
$$

Now as we approach with $$N$$ to infinity the fraction $$\frac{n_0}{N}$$ will approach to $$p(0)$$ and $$\frac{n_1}{N}$$ will approach to $$p(1)$$. So we can write

$$ \frac{n_0}{N}\log\frac{p(0)}{q(0)} + \frac{n_1}{N}\log\frac{p(1)}{q(1)} \quad \xrightarrow{N \to \infty} \quad p(0) \cdot \log\frac{p(0)}{q(0)} + p(1) \cdot \log\frac{p(1)}{q(1)}$$

Which is exactly **Kullback-Leibler (KL) Divergence**!

{: .slogan }
> The **KL Divergence** between the probability distributions $$p$$ and $$q$$ defined as
>
> $$D_{\text{KL}}(p \vert\vert q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}$$
>
> measures how "**surprised**" we would be if someone generated the samples from the probability distribution $$p$$ and said that they generated the samples from the probability distribution $$q$$. The lower the value of **KL Divergence** the more similar the distributions are.
> 
> {: .note }
> The term [Surprisal](https://en.wikipedia.org/wiki/Information_content) is a real thing even though more is used the term **Information content** or **Shannon information**.

When we now uderstand the definition of the **KL Divergence** can you guess why do we use it? Because in machine learning we are doing exactly the same thing! We sample the data from the distribution $$p_\text{data}(\boldsymbol{x}, y)$$ and we want to measure how **surprised** we would be if we generated the samples from the distribution $$p_\text{model}(\boldsymbol{x}, y \vert \boldsymbol{w})$$ which we somehow got. 

## Cross-Entropy Loss

We can now put our probability distributions into the **KL Divergence** $$D_{\text{KL}}(p\, \vert\vert \,p^*)$$.

$$
\begin{aligned}
&D_{\text{KL}}(p\, \vert\vert \,p^{*}) = \sum_{(\boldsymbol{x}, y)\backsim p_\text{data}} p^{*}(y,\boldsymbol{x}) \cdot \log \biggl(\frac{p^{*}(y,\boldsymbol{x})}{p(y,\boldsymbol{x}\,\vert\, \boldsymbol{w})}\biggr) = \\ \\
&= \sum_{(\boldsymbol{x}, y)\backsim p_\text{data}} p^{*}(y,\boldsymbol{x}) \cdot \biggl[\log p^{*}(y,\boldsymbol{x}) - \log\Bigl(p(y\,\vert\,\boldsymbol{x},\boldsymbol{w}) \cdot p(\boldsymbol{x})\Bigr)\biggr] = \\ \\
&= \underbrace{\sum_{(\boldsymbol{x}, y)}  p^{*}(y,\boldsymbol{x}) \cdot \log p^{*}(y,\boldsymbol{x})}_{\text{Doesn't depend on }\boldsymbol{w}} - \sum_{(\boldsymbol{x}, y)}  p^{*}(y,\boldsymbol{x}) \cdot \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w}) - \underbrace{\sum_{(\boldsymbol{x}, y)} p^{*}(y,\boldsymbol{x}) \cdot \log p(\boldsymbol{x})}_{\text{Doesn't depend on }\boldsymbol{w}} = \\ \\
&= - \sum_{(\boldsymbol{x}, y)\backsim p_\text{data}}  p^{*}(y,\boldsymbol{x}) \cdot \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})
\end{aligned}
$$

we proved that minimizing the **Cross-Entropy Loss** $$H(p^{*}, p)$$ is equivalent to minimizing the **KL Divergence** $$D_{\text{KL}}(p\, \vert\vert \,p^{*})$$. This is the reason why we use the **Cross-Entropy Loss** instead of some other loss function.

$$\arg\min_{\boldsymbol{w}} D_{\text{KL}}(p\, \vert\vert \,p^{*}) = \arg\min_{\boldsymbol{w}} - \sum_{(\boldsymbol{x}, y)} p^{*}(y,\boldsymbol{x}) \cdot \log p(y\,\vert\,\boldsymbol{x}, \boldsymbol{w}) = \arg\min_{\boldsymbol{w}} H(p^{*}, p)$$

Now, minimizing the **Cross-Entropy** directly is not possible, because we don't know the true distribution $$p(y,\boldsymbol{x})$$. but we can approximate it like this:

$$
\begin{aligned}
& \arg\min_{\boldsymbol{w}}\biggl( - \sum_{(\boldsymbol{x}, y)\backsim p_\text{data}}  p^{*}(y,\boldsymbol{x}) \cdot \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})\biggr) = \arg\min_{\boldsymbol{w}}\biggl( - \mathbb{E}_{(\boldsymbol{x}, y)\backsim p_\text{data}} [\log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})]\biggr) \approx \\ \\ 
&\approx \arg\min_{\boldsymbol{w}}\biggl(\frac{1}{N} \sum_{(\boldsymbol{x}, y)\,\backsim\,\mathcal{D}} - \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})\biggr) = \arg\min_{\boldsymbol{w}}\biggl(- \sum_{(\boldsymbol{x}, y)\,\backsim\,\mathcal{D}} \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})\biggr)
\end{aligned}
$$

where $$\mathcal{D}$$ is the dataset and $$N$$ is the number of samples in the dataset.

{: .definition }

>The **Cross-Entropy Loss** from observed data from the true distribution $$p^{*}(y\,\vert\,\boldsymbol{x})$$ is defined as
>
> $$- \frac{1}{N} \sum_{(\boldsymbol{x}, y)\,\backsim\,\mathcal{D}} \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})$$
>
> where $$\mathcal{D}$$ is the dataset and $$N$$ is the number of samples in the dataset.


### Maximum Likelihood Estimation (MLE)

We can also derive the **Cross-Entropy Loss** from the **Maximum Likelihood Estimation (MLE)**. When we assert that the true distribution $$p^{*}(y\,\vert\,\boldsymbol{x})$$ is one when with the true label and zero otherwise we end up

$$
\begin{aligned}
&\arg\min_{\boldsymbol{w}}\biggl(- \sum_{(\boldsymbol{x}, y)\backsim\mathcal{D}} \log p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})\biggr) = 
\arg\min_{\boldsymbol{w}}\biggl(-\log \prod_{(\boldsymbol{x}, y)\backsim\mathcal{D}} p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})\biggr) =  \\ \\
&= \arg\max_{\boldsymbol{w}}\biggl(\prod_{(\boldsymbol{x}, y)\backsim\mathcal{D}} p(y\,\vert\,\boldsymbol{x},\boldsymbol{w})\biggr)
\end{aligned}
$$

## Softmax Function

As we mentioned earlier, the output of the classifier is a vector of unnormalized scores $$\boldsymbol{s}$$ - logits. But here we talk about a probability distribution output. To normalize the logits and convert them to probabilities we use the softmax function $$\boldsymbol{\sigma}(\boldsymbol{s})$$. 

<br>
<div align="center">
    <img src="{{ site.baseurl }}/assets/images/statistical_model.png" width="600px"/>
</div>
<br>

{: .definition }
> The **Softmax Function** denoted $$\boldsymbol{\sigma}(\boldsymbol{s})$$ is defined as
> 
>$$\boldsymbol{\sigma}(\boldsymbol{s})_i = \frac{e^{s_i}}{\sum_{j}^{c} e^{s_j}}$$
>
> where $$c$$ is the length of the vector $$\boldsymbol{s}$$.

The softmax function takes the logits $$\boldsymbol{s}$$ as input and outputs a vector of probabilities that sum to one. The softmax function is used to convert the raw scores output by the classifier into probabilities. The probabilities represent the model's confidence in each class and are used to make predictions. 






