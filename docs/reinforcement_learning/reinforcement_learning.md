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

This page covers reinforcement learning **(RL)** concepts and algorithms. Some formulations in this document might be a bit informal, since the goal is to provide the natural explanation of the terms, rather than the strict formalism.

This page describes introduction to formalization of RL task and provides an explanation of terms like: RL task, RL algorithm, states, rewards, actions, value function, action-value function, return, advantage, policy.

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

Formally, a _RL_ task is oftem formularized as infitnite-horizon Markov decision process (MDP), defined by the tuple $\mathcal{S},\mathcal{A},p,r$, where $\mathcal{S}$ is a set of states, $\mathcal{A}$ is a set of actions, p is a transition function $p: \mathcal{S}\times\mathcal{A} \times \mathcal{S} \rightarrow [0,1] $. Reward function $r(s,a)$ assigns returns a value, after action $a$ was taken in a state $s$.

## TODO

- [ ] definition of RL task
  - [ ] What are states? Actions - discrete vs continous
  - [ ] Note about reward
- [ ] what is goal, RL formulation and RL algorithms
- [ ] Definition of Policy
- [ ] Definition of Value Function, Advantage function
- [ ] Sources
