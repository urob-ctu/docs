---
title: Reinforcement Learning
layout: default
has_children: true
nav_order: 5
mathjax: true
---

# Reinforcement Learning

{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## How to read these RL pages

This page covers reinforcement learning **(RL)** concepts and algorithms. Some formulations in this document might be a bit informal, since the goal is to provide the natural explanation of the terms, rather than keeping the strict formalism.

This page describes introduction to formalization of RL task and provides an explanation of terms like: *RL task, RL algorithm, states, rewards, actions, value function, action-value function, return, advantage, policy.*

The page [Road to PPO]({% link docs/reinforcement_learning/road_to_ppo.md %}) than use the mentioned terms to explain what is a [policy gradient](https://en.wikipedia.org/wiki/Policy_gradient_method) and how the popular [PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization) algorithm can be derived from it.

## RL task

We start with an introductory example, which mentions important terms from the world of RL.

{: .note}

> As an example, consider the problem of financial decision making over a person's lifetime.
> This problem can be formulated as a reinforcement learning task, where the goal is to maximize cumulative wealth.
>
> The **state** $s\in\mathcal{S}$ of the person might consist of the financial situation, qualifications, and age of the person.
> At each decision point, the person can choose from a set of **actions** $a\in\mathcal{A}$ , such as starting a job (providing immediate income), enrolling in a paid course to improve qualifications (incurring an immediate cost but potentially increasing future income), or making financial investments (with uncertain long-term returns).
> The person can also **exploit** - keep its current salary in a company or **explore** - decide starting its own business, where the resulting **rewards** $r_x,r_{x+1},r_{x+2}, ...$ are unsure.
> The reward function $r: \mathcal{S}\times\mathcal{A} \rightarrow \mathbb{R}$ assigns immediate rewards (based on the previously taken actions and state in which you are now).

{: .definition}

> Formally, a _RL_ task is oftem formularized as infitnite-horizon Markov decision process (MDP), defined by the tuple $\mathcal{S},\mathcal{A},p,r$, where $\mathcal{S}$ is a set of states, $\mathcal{A}$ is a set of actions, p is a transition function $p: \mathcal{S}\times\mathcal{A} \times \mathcal{S} \rightarrow [0,1] $. Reward function $r(s,a)$  returns a value, after action $a$ was taken in a state $s$.

We can now imagine that our agent roams in the enviroment picks actions and moves from one state to another. This picture is easier to imagine for discrete actions and states, but can be generalized to an environment with continous actions and spaces. **For our course, we only assume that time is discretized**,
 meaning that we have exact points, when a chosen action is applied. (for simulators in computers, we are often discretize time to some $\Delta t$)

If we track how our agent moves in the environment and what rewards he is receiving, we obtain a trajectory $\tau$. Trajectory is represented as a sequence $s_0,a_0,r_0,s_1,a_1,r_1,s_2,a_2,r_2,...$, where triplet $s_t,a_t,r_t$ represents one *step*.

 *Beware, that we use symbol $r$ as reward function. $r_t$ is then the reward (value) obtained in t-th step*

{: .definition}
> We can now define the return of the trajectory $R(\tau)$. To handle potentially infinite trajectories, we introduce a discount factor $\gamma \in [0, 1)$. The discounted return is:
>
> $R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t $
>
> For tasks that are guaranteed to terminate (episodic tasks), we can sum up to a final step $T$, and $\gamma$ can be 1. The discount factor prioritizes immediate rewards over distant ones and ensures the sum is finite.

### Policy and expected return

We should now describe how our actions are picked. We are aware, that all information are stored in current state (from definition of MDP). So we could be able set up a function, but for purposes seen later, we will treat it as a probability distribution. This distribution is called **Policy**, is noted as
$ \pi(a|s) $ and describes our behaviour in MDP.

{: .definition}

> We can then define probability of the trajectory as:
>
> $$
> p(\tau | \theta) = p(s_0)\pi_\theta(a_0 | s_0)p(s_1 | s_0, a_0) ... p(s_{T-1} | s_{T-2}, a_{T-2})\pi_\theta(a_{T-1} | s_{T-1})p(s_{T}|s_{T-1}, a_{T-1})
> $$

Since we are possibly dealing with a stochastic environment (remember the transition function), we are not seeking to find a policy that would one-time bring a maximal return, we are looking for a policy that maximizes the expected return.

{: .definition}
> We introduce expected return $J$ as
>
> $$J = \mathbb{E}_\tau R(\tau) = \int_\tau p(\tau | \pi) R(\tau)  d\tau$$

{: .definition}
>The task of the RL algorithm is then:
>
> $$
> \max_\pi J
> $$

{: .note}
> Note the difference between return $R$ and mean return $J$

## Real applications  

Before moving to another boring explanation of RL algortihm-realated definitions, we should provide a big picture overview and discuss how solving a problem via RL looks like.

In practice, someone approaches you with "I want you to make this robot go through an obstacle course", "I want you to make computer play League of Legends" or in our introductory case "I want to learn how to be rich in average case person". This is a problem.

Our job is then to formalize this problem as a RL-task: define what will be states, action and reward function (Multiple formulations can be made - do it as an excercise!).

The choice is crucial - We need to represent states that are easy to understand by a neural network ([Road to PPO explains]({% link docs/reinforcement_learning/road_to_ppo.md %}) why) and choose a reward function, that provides suitable feedback for agent to learn (not too sparse, but also not exploitable).

If wrong choices of state, actions or reward function are made, no RL-algorithm will help us.

Finally we have a reasonable RL formulation and we can apply an RL algorithm to it (typically we use an algorithm that was successful on similar tasks to ours,). Popular library of pre-implemented algorithms is [Stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/).
 We need to tune hyperparameters and hope, it will learn...

The following diagram summarizes the tweaks we can make when using RL in practice:

## RL algorithm-related definitions (value function,...)

## TODO

- [x] definition of RL task
  - [x] What are states? Actions - discrete vs continous
  - [x] Note about reward
- [x] Definition of Policy
- [ ] what is goal, RL formulation and RL algorithms
  - [ ] diagram
- [ ] Definition of Value Function, Advantage function
  - [ ] example RL task to compute value functions,...
- [ ] Sources
