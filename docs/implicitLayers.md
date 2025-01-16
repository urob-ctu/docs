---
title: Implicit layers
layout: default
has_children: false
nav_order: 3
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

{: .note }
>The explicit function might look something like this. 
>$$ y = g(x).$$ 
>In contrast, an implicit layer is defined by an equation such as:
>$$ f(x, \omega) = 0 $$
>The input for the function is $$ \omega $$ and the output optimal value $$ \boldsymbol{x}^{\star}$$.


## Types of implicit layers

In this article i will cover these types of implicit layers, as those are covered in the lectures.
- **Root finder**
- **Unconstrained optimizer**
- **Constrained optimizer**
- **ODE solver** 

## Backpropagation

Like any other layer, implicit layers must support gradient backpropagation for the network to learn. To achieve this, we rely on implicit differentiation. If the layer’s output $$\boldsymbol{\omega}$$ satisfies an equation of the form:

$$ f(\boldsymbol{x}, \omega) = 0 $$

Where x represents the input, we than calculate the gradient of the loss function with respect to x and the model's parameters as: 

$$
\frac{\partial x}{\partial \omega} = -\left(\frac{\partial \omega}{\partial f}\right)^{-1} \frac{\partial x}{\partial f}
$$

Fortunately, for our us there is autograd which can backpropagate automatically through those layers.


## Root finder

The root finder solves an equation of the form $$ f(x, \omega) = 0 $$, where $$ \boldsymbol{x} $$ is the input, $$ \boldsymbol{\omega} $$ is the output and $$ \boldsymbol{f} $$ defines the relationship. This layer can use variety of methods for finding the optimal solution. Two of the typical and well known methods use for solving the problem is Newton's method and fixed point method. 

## Unconstrained optimizer 

{: .definition}
> **Unconstrained optimizer**: and optimizer designed to find the extrema of a function either maximum or minimum without any constraints of the variable $$ \boldsymbol{x} $$.    
> $ f(\boldsymbol{x}, \omega) = 0$

This layer can be implemented using method such as decent method or Newton's method. 

## Constraint optimizer

As we wish to solve more complex problems we can use constraints for the optimizer.

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


In a more general form, we use the Karush Khun Tucker conditions (KKT).

Stationarity: $\nabla_x L = 0$   
Primal feasibility: $h_i(x) = 0, g_j(x)\le 0$    
Dual feasibility: $μ_j \ge  0$    
Complementary slackness: $μ_jg_j(x) = 0 $


## ODE solver

{: .definition}
>An **ODE (Ordinary Differential Equation) solver** integrates the solution of differential equation.    
> $ \dot{\boldsymbol{x}} = f(\boldsymbol{x}, \omega) $

This is a very useful way of modeling continuous time dynamics. Any solution obtained from a solver for the system must follow this conditions:    
$\dot{\boldsymbol{x}}^\star(t, \omega) - f(\boldsymbol{x}^\star(t, \omega), \omega) = 0 $


## Advantages of implicit layers
- **Flexibility**: Implicit layers can encode complex relationships, constraints and dynamics without requiring a predefined structure. 

- **Compact representation**: These layers cna model high-dimensional data or relationships with fewer parameters, leading to efficient models.

- **Adaptability**: Unlike explicit layers, implicit layers adapt to the input. 

## Challenges of implicit layers

- **Computational power**: For us to be able to use implicit layers we need a bit more computational power as some of the optimization problems can be challenging. 

- **Finding a solution**: For complex systems i can be hard to find the solution for the layer as the solution can be non-trivial. 

- **Gradient stability**: The differentiation is elegant but we have to be careful and be aware of any numerical instability. 


## Expected knowledge

From the text above you should understand the following concepts. 

- **Implicit layer**: How an implicit layer works and what does it mean. 
- **Types of implicit layers**: Basic understanding of the different types of implicit layers. 
- **Usage of implicit layers**: When and how to use the methods mentioned.

