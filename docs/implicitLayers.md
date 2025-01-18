---
title: Implicit layers
layout: default
has_children: false
nav_order: 5
mathjax: true
---

# Implicit Layers

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

Implicit layers are part of neural networks in the same way as any other layer. The difference lies in what these layers compute. For example, a sigmoid layer computes the output variable simply by applying a formula to the input variable. Implicit layers, however, rely on all variables involved in the equation.

## Motivation
Implicit layers unlock new fields for neural networks as they introduce new solutions to different problems. The power of solving optimization problems or differential equations can be used for modeling complex networks.

Using differential equations, we already build models for predicting forecasts or computing fluid dynamics.

One of the uses of constraint solvers is in robotics, as they can build trajectories within the given conditions.

In computer vision, differentiable layers enable tasks like 3D reconstruction and scene understanding.

## Implicit functions

{: .definition }
> An **implicit function** is a type of function where the relationship between the variables is given by an equation, rather than explicitly solving for one variable in terms of the others. In other words, the dependent variable is not isolated on one side of the equation. For example, $ f(\boldsymbol{x}, \omega) = \boldsymbol{x}^2 - \omega = 0$ 

{: .note}
> Some implicit functions can be rewritten explicitly under certain conditions, as rewriting can give us multiple solutions for specific cases.

For example, $ \boldsymbol{x} - \omega = 0 $ can be rewritten as $ \boldsymbol{x} = \omega $.   
A different problem is the implicit function $ \boldsymbol{x}^2 + \boldsymbol{y}^2 = 1$ (a circle in 2D plane and a hypersphere in multiple dimensions).    
From the graph, we can easily see that it cannot be expressed as one explicit function because there are multiple output values **y** for one input value **x**.  

**X** axis is horizontal and **Y** axis is vertical. 
<div align="center">
  <img src="{{ site.baseurl }}/assets/images/circle_in_2d.png" width="400">
</div>

However, we could use an explicit representation by adding constraints. In the first and second quadrants, we can represent the function as $ \boldsymbol{y} = \sqrt{1 - \boldsymbol{x}^2} $ where $ \boldsymbol{x} \in [-1, 1] $.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/half_circle.png" width="400">
</div>

A true example of an implicit function that cannot be expressed as explicit functions, even with conditions or piecewise definitions, is the following:

$ \sin{\boldsymbol{xy}} = \boldsymbol{y} $


## Root finder

{: .definition}
> A **Root finder** layer can be defined like this:    
> $ f(\boldsymbol{x}, \omega) = 0$    
> where we are solving for **x** based on the parameter $ \omega$.   
> The output of the layer is an optimal value $\boldsymbol{x^\star}$ for given $\omega$.

This layer can use a variety of methods to find the optimal solution. Two of the typical and well-known methods used for solving the problem are Newton's method and the Fixed Point method.

Like any other layer, a Root Finder layer has to be able to backpropagate the gradient of the loss to the weights in order to learn the network. We look for the gradient:

$ \frac{\partial \boldsymbol{x^{\star}}(\omega)}{\partial \omega}$.

We can derive an equation for the gradient from the derivative of the main function.

$ \frac{\partial f(\boldsymbol{x^\star} (\omega), \omega)}{\partial \omega} = \frac{\partial f(\boldsymbol{x^\star} (\omega), \omega)}{\partial \boldsymbol{x^\star}(\omega)} \cdot \frac{\partial \boldsymbol{x^{\star}}(\omega)}{\partial \omega} + \frac{\partial f(\boldsymbol{x^\star} (\omega), \omega)}{\partial \omega} = 0$

After we extract the derivative, we get:  

{: .definition}
>The final equation for **Implicit gradient**:    
>
>$\frac{\partial \boldsymbol{x^{\star}}(\omega)}{\partial \omega} = - [\frac{\partial f(\boldsymbol{x^\star} (\omega), \omega)}{\partial \boldsymbol{x^\star}(\omega)}]^{-1} \cdot \frac{\partial f(\boldsymbol{x^\star} (\omega), \omega)}{\partial \omega}$



### example
Let's take an implicit function $ \sin{\boldsymbol{x} + \omega} $. We can use this function in an implicit layer by finding the root of the function. After we obtain the root $ \boldsymbol{x^\star} $, we need to use the implicit gradient to backpropagate.

$ \frac{\partial \boldsymbol{x^{\star}}(\omega)}{\partial \omega} = -\frac{1}{\cos{(\boldsymbol{x^\star} + \omega)}} \cdot \cos{(\boldsymbol{x^\star} + \omega)} = -1 $

## Other types of implicit layers

Lectures explain other implicit layers beyond the Root Finder. This is a quick summary of those layers.

### Unconstrained optimizer 

{: .definition}
> **Unconstrained optimizer**: An optimizer designed to find the extrema of a function, either maximum or minimum, without any constraints on the variable $ \boldsymbol{x} $.    
> $ \arg{ \min{f(\boldsymbol{x}, \omega)}}$

This layer can be implemented using methods such as Gradient Descent or Newton's method.

### Constraint optimizer

For solving more complex problems, we can use constraints in the optimizer.

{: .definition}
>**Constrained optimization**
>
>$ \arg \min{f(\boldsymbol{x}, \omega)} $    
>Subject to:    
>   $ g(\boldsymbol{x}, \omega) \le 0$ (Inequality constraints)    
>   $ h(\boldsymbol{x}, \omega) = 0$ (Equality constraints)


To solve this problem we can use Lagrange multipliers. 

$ L(\boldsymbol{x}, \omega, λ,μ)=f(\boldsymbol{x}, \omega)+\sum{\lambda_i h_i(x)}+\sum{\mu_j g_j(x)} $   
$\mu_j \ge 0$


In a more general form, we use the Karush-Khun-Tucker (KKT)conditions.

Stationarity: $\nabla_x L = 0$   
Primal feasibility: $h_i(x) = 0, g_j(x)\le 0$    
Dual feasibility: $μ_j \ge  0$    
Complementary slackness: $μ_jg_j(x) = 0 $


### ODE solver

{: .definition}
>An **ODE (Ordinary Differential Equation) solver** integrates the solution of differential equation.    
> $ \dot{\boldsymbol{x}} = f(\boldsymbol{x}, \omega) $

This is a very useful way of modeling continuous-time dynamics. Any solution obtained from a solver for the system must follow these conditions:    
$\dot{\boldsymbol{x}}^\star(t, \omega) - f(\boldsymbol{x}^\star(t, \omega), \omega) = 0 $


## Advantages of implicit layers
- **Flexibility**: Implicit layers can encode complex relationships, constraints and dynamics without requiring a predefined structure. 

- **Compact representation**: These layers can model high-dimensional data or relationships with fewer parameters, leading to efficient models.

- **Adaptability**: Unlike explicit layers, implicit layers adapt to the input. 

## Challenges of implicit layers

- **Computational power**: For us to be able to use implicit layers we need a bit more computational power as some of the optimization problems can be challenging. 

- **Finding a solution**: For complex systems it can be hard to find the solution for the layer as the solution can be non-trivial. 

- **Gradient stability**: The differentiation is elegant, but we have to be careful and be aware of any numerical instability. 


## Expected knowledge

From the text above, you should understand the following concepts:

- **Implicit layer**: How an implicit layer works and what it means.
- **Types of implicit layers**: A basic understanding of the different types of implicit layers.
- **Usage of implicit layers**:  When and how to use the methods mentioned.
