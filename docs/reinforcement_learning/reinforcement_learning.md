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

At first, we provide introduction to formalization of RL task, and then we explain terms like: *RL task, RL algorithm, states, rewards, actions, value function, action-value function, return, advantage, policy.* Similar page providing this introduction is this[^1].

The page [Road to PPO]({% link docs/reinforcement_learning/road_to_ppo.md %}) then uses the mentioned terms to explain what is a [policy gradient](https://en.wikipedia.org/wiki/Policy_gradient_method) and how the popular [PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization) algorithm can be derived from it.

## RL task

We start with an introductory example, which mentions important terms from the world of RL.

{: .note}

> As an example, consider the problem of financial decision making over a person's lifetime.
> This problem can be formulated as a reinforcement learning task, where the goal is to maximize cumulative wealth.
>
> The **state** $s\in\mathcal{S}$ of the person might consist of the financial situation, qualifications, and age of the person.
> At each decision point, the person can choose from a set of **actions** $a\in\mathcal{A}$, such as starting a job (providing immediate income), enrolling in a paid course to improve qualifications (incurring an immediate cost but potentially increasing future income), or making financial investments (with uncertain long-term returns).
> The person can also **exploit** - keep its current salary in a company or **explore** - decide starting its own business, where the resulting **rewards** $r_x,r_{x+1},r_{x+2}, ...$ are unsure.
> The reward function $r: \mathcal{S}\times\mathcal{A} \rightarrow \mathbb{R}$ assigns immediate rewards (based on the previously taken actions and state in which you are now).

{: .definition}

> Formally, a *RL* task is oftem formularized as infitnite-horizon Markov decision process (MDP), defined by the tuple $\mathcal{S},\mathcal{A},p,r$, where $\mathcal{S}$ is a set of states, $\mathcal{A}$ is a set of actions, p is a transition function $p: \mathcal{S}\times\mathcal{A} \times \mathcal{S} \rightarrow [0,1] $. Reward function $r(s,a)$  returns a value, after action $a$ was taken in a state $s$.

We can now imagine that our agent roams in the enviroment picks actions and moves from one state to another. This picture is easier to imagine for discrete actions and states, but can be generalized to an environment with continous actions and spaces.

 As regards the time domain, **we assume that time is discretized**,
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

## RL algorithm-related definitions (value function, ...)

After introduction of  RL-task, we need to define a few terms that occurs during explanation of RL-algorithms.

{: .definition}
> A **value function** for a policy $\pi$ is
>
> $$ v_\pi:\mathcal{S} \rightarrow \mathbb{R} $$
>
> It is a function assigning each state $s$ the expected return
> $v_\pi(s)=\mathbb{E}_{\tau \sim \pi} [R(\tau)|s_0=s]$

Just remember that value function depends on policy.

Value function is our guide saying "this action pays you more in long term, you should use it more often". Except for trivial examples, we cannot compute value function directly, we can only approximate it.

If we would have finite and reasonable number of states, we could just store running mean for each state to obtain an estimator of Value function. In many ocasions this is not possible, so we learn a neural network which gets state and tries to predict value function.

Now we can define the action-value function. As the name says, it gives us what reward we can expect, if we take an action and then follow a policy,

{: .definition}
> An **Action-Value function** for a policy $\pi$ is
>
>$$Q_\pi(s,a) = \sum_{s'\in\mathcal{S}} p(s'|s,a)( r(s,a,s')+V_\pi(s')) $$
>

Action-Value function can give advice in style: "What return we get, if we keep everything as it is but make change in this action". This is important for us to come up with a better and better policy.

Last, but not least we have the Advantage function:

{: .definition}
> An **Advantage function** for a policy $\pi$ is
>
>$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$
>

We can interpret it as: "how much better or worse is this action in comparison with other actions on average".

### Finding the optimal policy

As it was stated before, the optimal policy must fullfill the requirement:

$$ \arg\max_\pi J $$

From this fact it is natural to expect that optimal policy has an optimal value function. This value function can be expressed in terms of optimal action value function:

$$ V^*(s) =  \max_a Q^*(s,a)$$

## Practice problem

<img src="{{ site.baseurl }}/assets/images/rl_problem.svg" width="90%">

1. Given the diagram and $\gamma=0.8$, calculate:
    1. $V^\pi(\textbf{s}_2)$
    2. $V^\pi(\textbf{s}_3)$
    3. $Q^\pi(\textbf{s}_1,a=1)$
    4. $Q^\pi(\textbf{s}_1,a=2)$
    5. $V^\pi(\textbf{s}_1)$
    6. $A^\pi(\textbf{s}_1,a=1)$
    7. $A^\pi(\textbf{s}_1,a=2)$
2. Assume $\gamma=0.8$, for this RL MDP compute (**hint**: start computation from last states):
    1. $Q^{\pi^*}$
    2. $V^{\pi^*}$
    3. $\pi^*$

<details collapse markdown="block"><summary><b>click to open/collapse the results</b></summary>

1. ---
    1. $$V^{\pi}({\bf s}_{2})=0.5[1.0(-1)]+0.5[0.6(-1)+0.4(1)]=-\,0.6$$
    2. $$V^{\pi}({\bf s}_{3})=0.9[1.0(1)]+0.1[1.0(-1)]=0.8$$
    3. $$Q^{\pi}({\textbf{s}_{1}},{\bf a}=1)=0.7[1+0.8V^{\pi}({\bf s_{2}})]+0.3[-1+0.8V^{\pi}({\bf s_{3}})]=0.256$$
    4. $$Q^{\pi}({\bf s}_{1},{\bf a}=2)=1.0[-1+0.8\,V^{\pi}({\bf s}_{3})]=-0.36$$
    5. $$V^{\pi}({\bf s}_{1})=0.8Q^{\pi}({\bf s}_{1},{\bf a}=1)+0.2Q^{\pi}({\bf s}_{1},{\bf a}=2)=0.1328 $$
    6. $$A^{\pi}({\bf s}_{1},{\bf a}=1)=Q^{\pi}({\bf s}_{1},{\bf a}=1)-V^{\pi}({\bf s}_{1})=0.1232$$
    7. $$A^{\pi}({\bf s}_{1},{\bf a}=2)=Q^{\pi}({\bf s}_{1},{\bf a}=2)-V^{\pi}({\bf s}_{1})=-0.4928$$
2. ---
1. $$Q^{\pi^{*}}({\bf s}_{2},{\bf a}=1)=1.0\cdot(-1+0.8\cdot0)=-1$$
2. $$Q^{\pi^{*}}(\mathbf{s}_{2},\mathbf{a}=2)=0.4\cdot(1+0.8\cdot0)+0.6\cdot(-1+0.8\cdot0)=-0.2$$
3. $$Q^{\pi^{*}}(\mathbf{s}_{3},\mathbf{a}=1)=1.0\cdot(1+0.8\cdot0)=1$$
4. $$Q^{\pi^{*}}(\mathbf{x}_{3},\mathbf{u}=2)=1.0\cdot(-1+0.8\cdot0)=-1$$
5. $$Q^{\pi^\ast}({\bf x}_{1},{\bf u}=1)=0.7\cdot(1+0.8\cdot(-0.2))+0.3\cdot(-1+0.8\cdot1)=0.528$$
6. $$Q^{\pi^{*}}({\bf x}_{1},{\bf u}=2)=1.0\cdot(-1+0.8\cdot1)=-\,0.2$$
7. $$V^{\pi^{*}}(\mathbf{x}_{2})=-\,0.2$$
8. $$V^{\pi^{*}}({\bf x}_{3})=1$$
9. $$V^{\pi^{*}}(\mathbf{x}_{1})=0.528$$
10. $$\pi^*(\bf s_i) =\begin{cases} \textbf{a}=1 \  \text{if} \ i=1 \\ \textbf{a}=2 \ \text{if} \ i=2 \\ \textbf{a}=1 \ \text{if} \ i=3     \end{cases}$$

</details>

## Real applications  

Now that you have initial intuition, what a formulation of RL-task consist of, we should provide a big picture overview and discuss how solving a problem via RL looks like.

In practice, someone approaches you with "I want you to make this robot go through an obstacle course", "I want you to make computer play League of Legends" or in our introductory case "I want to learn how to be rich in average case person". This is a RL problem.

Our job is then to formalize this problem as a RL-task: define what will be states, action and reward function (Multiple formulations can be made - do it as an excercise!).

The choice is crucial - We need to represent states that are easy to understand by a neural network ([Road to PPO explains]({% link docs/reinforcement_learning/road_to_ppo.md %}) why). Our states also have to satisfy markov property. The choice of a reward function is also important, we need a function that provides suitable feedback for agent to learn (not too sparse, but also not exploitable).

If wrong choices of state, actions or reward function are made, no RL-algorithm will help us.

Finally, we have a reasonable RL formulation and can apply an RL algorithm to it (typically we use an algorithm that was successful on similar tasks to ours). Popular library of pre-implemented algorithms is [Stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/).
 We need to tune hyperparameters and hope, it will learn...

The following diagram summarizes the tweaks we can make when using RL in practice:

<img src="{{ site.baseurl }}/assets/images/rl-task-result.svg" width="95%">

## TODO

* [x] definition of RL task
  * [x] What are states? Actions - discrete vs continous
  * [x] Note about reward
* [x] Definition of Policy
* [x] what is goal, RL formulation and RL algorithms
  * [x] diagram
* [x] Definition of Value Function, Advantage function
  * [x] action value function
* [x] example RL task to compute value functions,...
* [x] Sources

## References

[^1]: [Spinning up RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
