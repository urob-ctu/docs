---
title: Road to PPO
layout: default
parent: Reinforcement Learning
nav_order: 1
mathjax: true
---

# Road to PPO

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

The [previous page]({% link docs/reinforcement_learning/reinforcement_learning.md %}) introduced a RL task and informed us, that the ultimate goal of the RL algorithm is to find a policy $$\pi^*: \mathcal{S} \rightarrow \mathcal{A}$$, such that:

$$\pi^* = \arg \max_\pi J$$

where $$J$$ is return.
In this page we will continue with equations to derive the policy gradient: $$\nabla J $$. Later, we will come up with the PPO[^1] algorithm.

## Simplest Policy Gradient

Before we continue, we will assume that our policy $$ \pi $$ is:

- **parametrized**: We are using neural network with learnable parameters $$\theta$$. This network provides mapping: $$\mathcal{S} \rightarrow_\theta \mathcal{A}$$. From this point now, the symbol $\pi_\theta$ denotes policy parametrized by parameters $\theta$. This neural network is often called an *actor network*.
- **stochastic**: Instead of directly outputting the action, our neural network will output parameters for a probability distribution. As an example our network outputs $\mu_\theta$, $\sigma_\theta$ and this will be used in normal distribution $\mathcal{N}(\mu,\sigma)$.

We will now derive the approximation of gradient:
$$ \nabla_\theta J \approx \dfrac{1}{N} \sum_{i=1}^N \nabla_\theta \log (p(\tau_i | \theta)) R(\tau_i) = \dfrac{1}{N}\dfrac{1}{T} \sum_{i=1}^N R(\tau_i) \sum_{t=0}^{T-1} \nabla_\theta \log (\pi_\theta(a^i_t | s^i_t)) $$
where $R(\tau_i)$ is the return of the ith trajectory.

<details open markdown="block"><summary><b>click to open/collapse the proof</b></summary>

{: .proof}
> Following the source[^2]. The maximization objective is
>
>$$
>\mathbb{E}_\tau[R(\tau)|\theta] = \int_\tau p(\tau | \theta) R(\tau) d\tau
>$$
>
>$$\begin{align*}
>\nabla_{\theta} J &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}{R(\tau)} & \\
>&= \nabla_{\theta} \int_{\tau} p(\tau|\theta) R(\tau) & \text{Expand expectation} \\
>&= \int_{\tau} \nabla_{\theta} p(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
>&= \int_{\tau} p(\tau|\theta) \nabla_{\theta} \log p(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
>&= \mathbb{E}_{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log p(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
> &= \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta} (a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob} \\
> &\approx  \dfrac{1}{N}\dfrac{1}{T} \sum_{i=1}^N R(\tau_i) \sum_{t=0}^{T-1} \nabla_\theta \log (\pi_\theta(a^i_t | s^i_t)) & \text{Estimatation via sample mean}
>\end{align*}$$
>
>In the derivation a few tricks were used:
> - **Probability of the trajectory**: The probability of a trajectory $\tau = (s_0, a_0, ..., s_{T+1})$ given that actions  come from $\pi_{\theta}$ is
> $$ p(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} p(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t) $$
> where $\rho_0 (s_0)$ is the probability, that the initial state is $s_0$. Applying logarithm to both sides, we obtain:  
> $$\log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg)$$
> and taking gradient operation...
>
>$$ \nabla_\theta \log p(\tau | \theta) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)  $$
>
> - **Log-Derivative trick**: The log-derivative trick is based on a simple rule from calculus: the derivative of $\log x$ with respect to $x$ is $1/x$. When rearranged and combined with chain rule, we get:
>
> $$ \nabla_{\theta} p(\tau | \theta) = p(\tau | \theta) \nabla_{\theta} \log p(\tau | \theta) $$
>
</details>

Nevertheless, if we use directly this approximation for a RL problem, the results will not be satisfying. Our estimation is unbiased, but high variance causes instability, making learning impractically slow. Our next steps will go towards the goal of variance reduction

## Introducing rewards-to-go

The current formula
$$\mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta} (a_t |s_t) R(\tau)}$$ takes into account whole reward of the trajectory, but this does not make much sense. Imagine a trajectory where at the first half suboptimal actions are performed and in the second half really great actions are taken. The reward of this trajectory will reinforce all the actions of the trajectory.

But since we are dealing with MDP, the chosen action affects only rewards obtained after performing this action.
We edit formula for our gradient to the form:

$$ \nabla_{\theta} J = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})} $$

This form is also justified mathematically. It can be proven that rewards taken before action has zero mean, but non-zero variance. The reward-to-go formula is still unbiased, but with lower variance.  

## Introducing discount factor

Another effect that we can take into account is the fact, that rewards occuring far in the future are highly variable and weakly correlated with current actions. Due to this fact, we can introduce a discount factor $\gamma \in (0,1)$ and compute rewards of trajectory as:

$$ R(\tau) = \sum_{t=0}^{\infty} \gamma^{t} r_t $$

Plugin this intro our previous estimator, we obtain:

$$ \nabla_{\theta} J_\gamma = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}, s_{t'+1})}  $$

Nevertheless, this estimator is biased, we are no longer converging to the true objective
$$
 J= \mathbb{E}_{\tau}[\sum^{T-1}_{t=0}r_t]
 $$, but rather the discounted objective $$
 J_\gamma= \mathbb{E}_{\tau}[\sum^{T-1}_{t=0}  \gamma^t r_t]
 $$. Thus

 $$
\nabla_\theta J \neq \nabla_\theta J_\gamma
 $$

The closer is $\gamma$ to zero, the more we are converging to a policy that prefers immediate rewards (better to rob a bank now, then to gradually invest...)

## Subtracting the baseline

In this section, we will show that subtracting any baseline function $b(s_t)$ depending only on states, does not change the policy gradient $\nabla_\theta J$.

For this approach we use the fact known as EGLP lemma:

$$
\mathbb{E}*{a_t \sim \pi*\theta}[\nabla_\theta \log(P_\theta(x))]=0
$$

<details collapse markdown="block"><summary><b>click to open/collapse the proof</b></summary>

{: .proof}
> Approach based on **openAI spinning up**[^2].
>
> one can observe the fact:
>
> $$
> \nabla_\theta \int_{a_t} \pi_\theta(a_t|s_t) = \nabla_\theta 1 = 0
> $$
>
> According to the Leibniz rule,
>
> $$
> \nabla_\theta \int_{a_t} \pi_\theta(a_t|s_t) =  \int_{a_t}  \nabla_\theta \pi_\theta(a_t|s_t)
> $$
>
> Using the log-trick
>
> $$
> 0 = \int_{a_t}  \nabla_\theta \pi_\theta(a_t|s_t) = \int_{a_t} \pi_\theta(a_t|s_t) \nabla_\theta \log(\pi_\theta(a_t|s_t)) = \mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log(P_\theta(x))]
> $$
</details>

Now using the known fact about mean value

$$
\mathbb{E}*{a_t \sim \pi*\theta}[\nabla_\theta \log(P_\theta(x)) b(s_t)] = b(s_t) \mathbb{E}*{a_t \sim \pi*\theta}[\nabla_\theta \log(P_\theta(x))] = b(s_t) \cdot 0 = 0
$$

for arbitrary function $b$ which only depends on state.
This allows us to add or subtract any number of terms like this from our expression for the policy gradient, without changing it in expectation:

$$
\nabla_{\theta} J = \mathbb{E}*{\tau \sim \pi*{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left((\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})) - b(s_t)\right)}.
$$

The common choice of baseline is on-policy value function $b(s_t) = V^\pi(s_t) $
We are unable to have the *real* value function (otherwise it would be a solution to whole RL problem), so we use approximation of it.
Usually, a neural network $V_\phi(s_t)$ is used (often called *critic network*).

This value (critic) network is trained in parallel with the policy to regress value targets $V^*(s)$, which are estimated from the trajectory rewards. This is typically done by minimizing the L2 distance between the value network and the value targets

$$
L_v(\theta) = \dfrac{1}{N}\dfrac{1}{T} \sum_{i=1}^N \sum_{t=0}^{T-1}  \left( \hat{V}_\theta(s_t^i) - V^*(s_t^i)\right)^2
$$


The architecture pf actor and critic networks can be totally isolated or share some common layers.


## TODO

- [ ] add GIF of inverse pendulum for vanilla policy-grad and its improvements
- [ ] Leibniz rule

## References

[^1]: Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
[^2]: [Spinning up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
