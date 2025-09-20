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

> Consider the problem of financial decision making over a person's lifetime.
> This problem can be formulated as a reinforcement learning task, where the goal is to maximize cumulative wealth.
>
> The **state** of the person might consist of the financial situation, qualifications, and age of the person.
> At each decision point, the person can choose from a set of **actions**, such as starting a job (providing immediate income), enrolling in a paid course to improve qualifications (incurring an immediate cost but potentially increasing future income), or making financial investments (with uncertain long-term returns).
> The person can also **exploit** - keep its current salary in a company or **explore** - decide starting its own business, where the resulting **reward** is unsure.
> The reward function assigns immediate rewards based on the chosen action, such as receiving a salary or incurring a cost, while also considering long-term rewards, such as increased earning potential or financial stability.
>
> This example highlights the trade-off between immediate and delayed rewards, a fundamental aspect of reinforcement learning. The person must balance short-term gains with long-term benefits.

## TODO

- [ ] definition of RL task
  - [ ] What are states? Actions - discrete vs continous
  - [ ] Note about reward
- [ ] what is goal, RL formulation and RL algorithms
- [ ] Definition of Policy
- [ ] Definition of Value Function, Advantage function
- [ ] Sources
