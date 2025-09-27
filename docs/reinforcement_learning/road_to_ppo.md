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

$$\pi^* = \arg \max_\pi J_\pi$$

where $$J_\pi$$ is an expected return of policy $\pi$.
In this page we will continue with equations to derive the policy gradient: $$\nabla J_\pi $$. Later, we will come up with the PPO[^1] algorithm.

## Simplest Policy Gradient

Before we continue, we will assume that our policy $$ \pi $$ is:

- **parametrized**: We are using neural network with learnable parameters $$\theta$$. This network provides mapping: $$\mathcal{S} \rightarrow_\theta \mathcal{A}$$. From this point now, the symbol $\pi_\theta$ denotes policy parametrized by parameters $\theta$. This neural network is often called an _actor network_.
- **stochastic**: Instead of directly outputting the action, our neural network will output parameters for a probability distribution. As an example our network outputs $\mu_\theta$, $\sigma_\theta$ and this will be used in normal distribution $\mathcal{N}(\mu,\sigma)$.

{: .warning}
> From this point we will write expected return just as $J$ and we interpret it as a function of weights $\theta$

We will now derive the approximation of gradient:
$$ \nabla_\theta J \approx \dfrac{1}{N} \sum_{i=1}^N \nabla_\theta \log (p(\tau_i | \theta)) R(\tau_i) = \dfrac{1}{N}\dfrac{1}{T} \sum_{i=1}^N R(\tau_i) \sum_{t=0}^{T-1} \nabla_\theta \log (\pi_\theta(a^i_t | s^i_t)) $$
where $R(\tau_i)$ is the return of the ith trajectory.

<details open markdown="block"><summary><b>click to open/collapse the proof</b></summary>

{: .proof}

> Following the source[^2]. The maximization objective is
>
> $$
> \mathbb{E}_\tau[R(\tau)|\theta] = \int_\tau p(\tau | \theta) R(\tau) d\tau
> $$
>
> $$
> \begin{align*}
> \nabla_{\theta} J &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}{R(\tau)} & \\
> &= \nabla_{\theta} \int_{\tau} p(\tau|\theta) R(\tau) & \text{Expand expectation} \\
> &= \int_{\tau} \nabla_{\theta} p(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
> &= \int_{\tau} p(\tau|\theta) \nabla_{\theta} \log p(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
> &= \mathbb{E}_{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log p(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
> &= \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta} (a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob} \\
> &\approx  \dfrac{1}{N}\dfrac{1}{T} \sum_{i=1}^N R(\tau_i) \sum_{t=0}^{T-1} \nabla_\theta \log (\pi_\theta(a^i_t | s^i_t)) & \text{Estimatation via sample mean}
> \end{align*}
> $$
>
> In the derivation a few tricks were used:
>
> - **Probability of the trajectory**: The probability of a trajectory $\tau = (s_0, a_0, ..., s_{T+1})$ given that actions come from $\pi_{\theta}$ is
>   $$ p(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} p(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t) $$
> where $\rho_0 (s_0)$ is the probability, that the initial state is $s_0$. Applying logarithm to both sides, we obtain:  
> $$\log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t) + \log \pi_{\theta}(a_t |s_t)\bigg)$$
>   and taking gradient operation...
>
> $$ \nabla_\theta \log p(\tau | \theta) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) $$
>
> - **Log-Derivative trick**: The log-derivative trick is based on a simple rule from calculus: the derivative of $\log x$ with respect to $x$ is $1/x$. When rearranged and combined with chain rule, we get:
>
> $$ \nabla_{\theta} p(\tau | \theta) = p(\tau | \theta) \nabla_{\theta} \log p(\tau | \theta) $$

</details>

Nevertheless, if we use directly this approximation for a RL problem, the results will not be satisfying. Our estimation is unbiased, but high variance causes instability, making learning impractically slow. Our next steps will go towards the goal of variance reduction

## Introducing rewards-to-go

The current formula
$$\mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta} (a_t |s_t) R(\tau)}$$ takes into account whole reward of the trajectory, but this does not make much sense. Imagine a trajectory where at the first half suboptimal actions are performed and in the second half really great actions are taken. The reward of this trajectory will reinforce all the actions of the trajectory.

But since we are dealing with MDP, the chosen action affects only rewards obtained after performing this action.
We edit formula for our gradient to the form:

$$ \nabla_{\theta} J = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s\_{t'+1})} $$

This form is also justified mathematically. It can be proven that rewards taken before action has zero mean, but non-zero variance. The reward-to-go formula is still unbiased, but with lower variance.

## Introducing discount factor

Another effect that we can take into account is the fact, that rewards occuring far in the future are highly variable and weakly correlated with current actions. Due to this fact, we can introduce a discount factor $\gamma \in (0,1)$ and compute rewards of trajectory as:

$$ R(\tau) = \sum\_{t=0}^{\infty} \gamma^{t} r_t $$

Plugin this intro our previous estimator, we obtain:

$$ \nabla_{\theta} J_\gamma = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}, s\_{t'+1})} $$

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

$$ \mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log(P_\theta(x))]=0 $$

<details collapse markdown="block"><summary><b>click to open/collapse the proof</b></summary>

{: .proof}
> Approach based on **openAI spinning up**[^2].
>
> one can observe the fact:
>
>
$$

> \nabla_\theta \int_{a_t} \pi_\theta(a_t|s_t) = \nabla_\theta 1 = 0
>
> $$
>
> According to the Leibniz rule,
>
>
> $$
>
> \nabla_\theta \int_{a_t} \pi_\theta(a_t|s_t) = \int_{a_t} \nabla_\theta \pi\_\theta(a_t|s_t)
>
> $$
>
> Using the log-trick
>
>
> $$
>
> 0 = \int_{a_t} \nabla_\theta \pi_\theta(a_t|s_t) = \int_{a_t} \pi_\theta(a_t|s_t) \nabla_\theta \log(\pi_\theta(a_t|s_t)) = \mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log(P_\theta(x))]
> $$

</details>

Now using the known fact about mean value

$$
\mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log(P_\theta(x)) b(s_t)] = b(s_t) \mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log(P_\theta(x))] = b(s_t) \cdot 0 = 0
$$

for arbitrary function $b$ which only depends on state.
This allows us to add or subtract any number of terms like this from our expression for the policy gradient, without changing it in expectation:

$$
\nabla_{\theta} J = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left((\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})) - b(s_t)\right)}.
$$

The common choice of baseline is **on-policy value function** $b(s_t) = V^\pi(s_t) $
We are unable to have the *real* value function (otherwise it would be a solution to whole RL problem), so we use approximation of it.
Usually, a neural network $V_\phi(s_t)$ is used (often called _critic network_). The part of the formula $$((\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})) - b(s_t))$$ is estimate of the advantage function $A^\pi(s_t,a_t)$

This value (critic) network is trained in parallel with the policy to regress value targets $V^_(s)$, which are estimated from the trajectory rewards. This is typically done by minimizing the L2 distance between the value network and the value targets

$$
L_v(\theta) = \dfrac{1}{N}\dfrac{1}{T} \sum_{i=1}^N \sum_{t=0}^{T-1}  \left( \hat{V}_\phi(s_t^i) - V^*(s_t^i)\right)^2
$$

The architecture of actor and critic networks can be totally isolated ($\phi$ and $\theta$ do not overlap) or share some common layers. Do not forget, that when discount factor $\gamma$ is used, the network learns to predict already the discounted rewards.

## Idea of bootstraping

For feeding the rewards of trajectories into our _critic_ network, two options are available:

- We feed only discounted trajectory rewards: for value target $V^*(s_t)$ we feed $r_t + \gamma r_{t+1}+ \gamma^2 r_{t+2} + \dots + r_{T-1} $

This means that for last state in the trajectory, we are trying to learn to regress to $V^_(s_{T-1})=r_{T-1}$ - only one reward, this brings high variance, so netowrk is hard to learn!

- To mitigate this issue we append at the end of trajectory our estimate of value function from the terminal state: $$r_t + \gamma r_{t+1}+ \gamma^2 r_{t+2} + \dots + (r_{T-1} + \gamma \hat{V}_\phi (s_{T-1}) )$$. **This idea is called bootstrapping**.

If our estimate of value function is good, the bootstraping reduces variance. Nevertheless, this sword is double-edge. For bad estimate, we are introducing bias.

## Generalized advantage estimate (Optional)

The article[^3] introduces another hyperpameter for tuning $\lambda$. This parameter gives us a way to mix estimation between TD(0) and TD(1).

- If $\lambda=0$ we estimate:
  $$A^\pi(s_t,a_t) = r_t + \gamma \hat{V}_\phi(s_{t+1})- \hat{V}_\phi(s_t)$$.
  Meaning "I trust more to my estimate than to my current experience". This is nice at the end of the learning, if good value estimate is present. Otherwise, we are introducing bias.
- If $\lambda=1$ we estimate:
  $$A^\pi(s_t,a_t) = (\sum_{l=0}^{T-1} \gamma^l (r_{t+l} + \gamma \hat{V}_\phi (s_{T-1}))) - V(s_t)$$.
  As one can observe with this setting it is the same as learning the Critic network with enabled bootstraping.

Further explanation of GAE is in [this article](https://arxiv.org/abs/1506.02438).

## Proximal Policy Optimization

One major issue in vanilla policy gradients is that a too-large update can catastrophically degrade performance. If we change the policy too much in one step, the new policy might behave very differently (even randomly), causing the agent to “fall off a cliff” in terms of reward. Due to this fact, we want to make only "safe steps".

For further parts, let us define:

$$
r_\theta(a_t | s_t) = \dfrac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

which is simply the ratio of the 'new' probabilities
$\pi_\theta(a_t | s_t)$
divided by the 'old' probabilities $\pi_{\theta_{old}}(a_t | s_t)$. The 'old' probabilities correspond to $\theta_{old}$, which are the policy parameters before the gradient update. The reason for these additional terms is that we will keep track of the 'old' parameters $\theta_{old}$ in order to enforce the proximity of the policy behaviour.

### TRPO (optional)

The problem described above is a motivation behind **Trust Region Policy Optimization (TRPO)** (Predecessor of PPO, full paper can be found [here](https://arxiv.org/pdf/1502.05477)). TRPO tries to solve this optimization task:

$$
\max_{\theta} \; \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}, \, a \sim \pi_{\theta_{\text{old}}}}
\left[ r_\theta(a_t |s_t) \, \hat{A}(s,a) \right]
$$

subject to the trust region constraint:

$$
\mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}}
\left[ \text{KL}\!\left(\pi_{\theta_{\text{old}}}(\cdot|s) \,\|\, \pi_{\theta}(\cdot|s)\right) \right]
\leq \delta
$$

where:

- $\rho_{\pi_{\text{old}}}$ is the state visitation distribution under the old policy,
- $\hat{A}(s,a)$ is the estimated advantage function,
- $\delta$ is a small positive constant bounding the KL divergence.

The objective function is a lower bound on policy return, improving it inside thes bounds provide impeovement of policy return.

### PPO

Nevertheless, TRPO problem is a constrained optimization problem, so we would be unable to solve it with standard gradient descent. To overcome this issue, the same team that was behind TRPO reformulated the problem as PPO - **Proximal Policy optimization** (paper can be found [here](https://arxiv.org/abs/1707.06347)).

As we can see $r_\theta$ is simply the ratio of the 'new' probabilities $\pi_\theta(a_t | s_t)$ divided by the 'old' probabilities $\pi_{\theta_{old}}(a_t | s_t)$. The 'old' probabilities correspond to $\theta_{old}$, which are the policy parameters before the gradient update. The reason for these additional terms is that we will keep track of the 'old' parameters $\theta_{old}$ in order to enforce the proximity of the policy behaviour.
Note that

$$
\nabla_\theta r_\theta(a_t | s_t) |_{\theta=\theta_{old}} = \nabla_\theta \log(\pi_\theta(a_t | s_t)) |_{\theta=\theta_{old}}
$$

i.e. taking the gradient of this ratio is equivalent to taking the gradient of the policy log-probabilities when done at the initial parameters $\theta_{old}$. This gives rise to the following equivalent objective for the advantage-based policy gradient (here omitting the mean over all states and actions)

$$
J_{A}(a_t, s_t) =  \hat{A} \cdot r_\theta(a_t | s_t)
$$

where $\hat{A}$ is short for $\hat{A}(s_t, a_t)$ - the advantage estimate for state $s_t$ and action $a_t$. If we were to take the gradient of this objective, we would get exactly the same expression we used in policy gradient methods. The PPO replaces this by the following surrogate objective

$$
J_{PPO}(a_t, s_t) = \min \left( \hat{A} \cdot r_\theta(a_t | s_t), \quad \hat{A} \cdot \text{clip}(r_\theta(a_t | s_t), 1-\epsilon, 1+\epsilon) \right)
$$

where $ \epsilon$ is a hyperparameter known as a "clipping ratio". This objective essentially implements gradient clipping, but in the probability space. To be clear, this objective needs to be again averaged over all states and actions in the collected trajectories. The resulting policy gradient is

$$

\nabla_{\theta} J_{PPO} = \frac{1}{N}\frac{1}{T} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \min \left( \hat{A} \cdot r_\theta(a_t | s_t), \quad \hat{A} \cdot \text{clip}(r_\theta(a_t | s_t), 1-\epsilon, 1+\epsilon) \right)

$$

### The idea behind PPO

To understand, let us see what happens, when the advantage is positive $\hat{A} > 0$, i.e. we would like to increase the probability of the given action $a_t$. Then the objective reduces to

$$
\hat{A} \cdot \min \left(  r_\theta(a_t | s_t), \quad \text{clip}(r_\theta(a_t | s_t), 1-\epsilon, 1+\epsilon) \right)
$$

Now, if
$r_\theta(a_t | s_t) \leq 1+\epsilon$
, then the $\min()$ chooses the ratio itself and the result is the same as if we used the original objective. But if $r_\theta(a_t | s_t) > 1+\epsilon$, then the objective is stuck at the constant value of $\hat{A} \cdot (1+\epsilon)$ and therefore we get zero gradient. This means that the objective allows the policy to increase the probability of the action $a_t$ by at most $(1+\epsilon)$-times the 'old' probability.

Similarly, for a negative advantage $\hat{A} < 0$, we would like to reduce the probability of $a_t$. The PPO objective would be

$$
\hat{A} \cdot \max \left(  r_\theta(a_t | s_t), \quad \text{clip}(r_\theta(a_t | s_t), 1-\epsilon, 1+\epsilon) \right)
$$

If
$r_\theta(a_t | s_t) \geq 1-\epsilon$, then the $\max$ chooses the ratio itself and nothing interesting happens. If $r_\theta(a_t | s_t) < 1-\epsilon$, however, then the objective is again stuck at a constant value of $\hat{A} \cdot (1-\epsilon)$ and the gradient is zero again. This does not let the probability of $a_t$ fall below $(1-\epsilon)$-times the 'old' probability.

So, essentially, the PPO objective will not allow the probabilities of the actions to change too much with respect to the 'old' policy that was used to sample the trajectories.

The following image illustrates the clipping objective for positive and negative advantages respectively.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*RUEQ7RXzywlV63nZ0ldJUg.png" width="600" align="center"/>

## TODO

- [ ] add GIF of inverse pendulum for vanilla policy-grad and its improvements
- [ ] Leibniz rule
- [ ] drawing explaining bootstrapping

## References

[^1]: Schulman, J., et al. (2017). _Proximal Policy Optimization Algorithms_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
[^2]: [Spinning up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
[^3]: [GAE](https://arxiv.org/abs/1506.02438)
