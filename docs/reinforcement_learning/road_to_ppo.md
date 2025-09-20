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
In this page we will continue with equations to derive the policy gradient: $$\nabla J $$

Before we continue, we have to assume that our policy $$ \pi $$ is:

- **parametrized**: We are using neural network with learnable parameters $$\theta$$. This network provides mapping: $$\mathcal{S} \rightarrow_\theta \mathcal{A}$$. From this point now, the symbol $\pi_\theta$ denotes policy parametrized by parameters $\theta$.
- **stochastic**: Instead of directly outputting the action, our neural network will output parameters for a probability distribution. As an example our network outputs $\mu_\theta$, $\sigma_\theta$ and this will be used in normal distribution $\mathcal{N}(\mu,\sigma)$.

  {: .proof}

> Pbbb
> ASDAS
