---
title: Optimization
layout: default
nav_order: 5
parent: Training
mathjax: true
---

## Optimization: Convergence Rate, Oscillations, and Diminishing Gradients

Today we'll explore how to find the minimum of a function-the fundamental problem at the heart of training machine learning models. We focus on **unconstrained optimization**, specifically finding $\min f(x)$, and discuss two main approaches: **Newton's Method** and **Gradient Descent**.

---

## Iterative Optimization Fundamentals

In optimization, we start at a point $x_k$ and iteratively search for a better point $x_{k+1}$ that brings us closer to the minimum $x^*$. Each iteration updates our position by moving in a search direction $d$:

$$x_{k+1} = x_k + d$$

The key question is: **How do we choose the direction $d$ and how far do we go (the step size $\alpha$)?**


### Function Approximation

To choose an effective direction, we approximate the function $f(x)$ locally around our current point $x_k$. The simplest approach is a **linear approximation** based on the **gradient** (first derivative).

**Linear Approximation:** We minimize the function's linear approximation, which is computationally straightforward. However, this only provides the **best direction** (the negative gradient), not the **optimal step size**.

The step size $\alpha$ (also called the **learning rate**) determines how far we move in the chosen direction:

$$x_{k+1} = x_k - \alpha \nabla f(x_k)$$

---

## Newton's Method: Using Second-Order Information

Newton's method provides a potentially **optimal step size** by using a more accurate approximation: the **second-order Taylor polynomial**.

### Second-Order Taylor Approximation

By incorporating the second derivative (or the **Hessian matrix $H$** in higher dimensions), we obtain a much better local approximation. The Taylor series expansion around $x_k$ becomes:

$$f(x) \approx f(x_k) + f'(x_k)(x - x_k) + \frac{1}{2}f''(x_k)(x - x_k)^2$$

This **quadratic approximation** captures the curvature of the function, not just its slope.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/newthon.png" width="800">
</div>

### Deriving the Newton Step

To find the minimum of this quadratic approximation $g(x)$, we set its derivative to zero:

$$g'(x) = f'(x_k) + f''(x_k)(x - x_k)$$

Setting $g'(x_{k+1}) = 0$ and solving for $x_{k+1}$:

$$f'(x_k) + f''(x_k)(x_{k+1} - x_k) = 0$$

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

Here $$\alpha = \frac{1}{f''(x_k)}$$

**Multivariate Case:** For vector-valued $x$, this generalizes to:

$$x_{k+1} = x_k - H^{-1}(x_k) \cdot \nabla f(x_k)$$

Where:
- $\nabla f(x_k)$ is the **gradient vector** (first derivatives)
- $H(x_k)$ is the **Hessian matrix** (second partial derivatives)
- $H^{-1}(x_k)$ is the inverse of the Hessian

Note that the Newton step effectively multiplies the gradient by $H^{-1}$, which automatically adjusts the step size based on the function's curvature.

### Advantages: Quadratic Convergence

Newton's method exhibits **quadratic convergence**, meaning the error decreases quadratically with each iteration. This results in very rapid convergence to the minimum, often requiring only a few iterations.

### The Computational Bottleneck

Despite its fast convergence, Newton's method is impractical for neural networks due to the **Hessian matrix**:

1. **Computing $H$:** Requires $O(N^2)$ operations for $N$ parameters (millions in modern networks)
2. **Inverting $H$:** Requires $O(N^3)$ operations

These computational costs make Newton's method infeasible for high-dimensional optimization in deep learning, leading us to the more practical **Gradient Descent**.

---

## Gradient Descent

Gradient Descent simplifies Newton's method by ignoring second-order information (the Hessian) and using only the gradient direction with a fixed step size.

### The Update Rule

Gradient descent uses a manually chosen **step size** $\alpha$ (the **learning rate**) to move in the direction of steepest descent:

$$x_{k+1} = x_k - \alpha \cdot \nabla f(x_k)$$

The choice of $\alpha$ is critical-it directly impacts convergence speed and stability.

### Convergence Analysis in 1D

Consider a simple quadratic function:

$$f(w) = 2w^2$$

The gradient is $\frac{\partial f}{\partial w} = 4w$, so the update rule becomes:

$$w_1 = w_0 - \alpha(4w_0) = w_0(1 - 4\alpha)$$

Applying this iteratively:

$$w_k = (1 - 4\alpha)^k w_0$$

**Convergence Condition:** For $w_k \rightarrow 0$ as $k \rightarrow \infty$, we need:

$$|1 - 4\alpha| < 1 \quad \Rightarrow \quad 0 < \alpha < \frac{1}{2}$$

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/convergence_1d.png" width="800">
</div>

**Convergence Rate:** The convergence rate determines how quickly we reach the minimum and is defined as:

$$r(\alpha) = |1 - 4\alpha|$$

- **Too small** ($\alpha < 1/4$): $r(\alpha)$ is close to 1 → **slow convergence**
- **Optimal** ($\alpha = 1/4$): $r(\alpha) = 0$ → **fastest convergence**  
- **Too large** ($\alpha > 1/2$): $r(\alpha) > 1$ → **oscillation and divergence**



### Convergence Analysis in 2D

The challenge becomes more apparent in multiple dimensions. Consider a quadratic function with different curvatures in each direction:

$$f(w) = \frac{1}{2}(w_1^2 + 4w_2^2) = \frac{1}{2} \begin{bmatrix} w_1 & w_2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}$$

The gradient is $\nabla f = \begin{bmatrix} w_1 \\ 4w_2 \end{bmatrix}$, giving us decoupled updates:

$$w_1^{(k)} = w_1^{(0)}(1 - \alpha)^k$$
$$w_2^{(k)} = w_2^{(0)}(1 - 4\alpha)^k$$

**Key Insight:** The eigenvalues of the Hessian are $\lambda_1 = 1$ and $\lambda_2 = 4$. Each dimension converges at rate $\vert 1 - \alpha\lambda_i \vert$.

<div align="center">
  <img src="{{ site.baseurl }}/assets/images/convergence_2d.png" width="800">
</div>

**The Fundamental Problem:** With a single learning rate $\alpha$ for all dimensions, the **overall convergence rate** is limited by the **slowest component**:

$$r(\alpha) = \max\{|1 - \alpha|, |1 - 4\alpha|\}$$

- If $\alpha < 1/4$: The $w_2$ direction (with larger eigenvalue $\lambda_2=4$) converges slowly
- If $\alpha > 1/2$: The $w_2$ direction oscillates and may diverge

**Ill-Conditioning:** When eigenvalues differ significantly ($\lambda_{\max} \gg \lambda_{\min}$), we face a dilemma: a learning rate suitable for one dimension is suboptimal (or unstable) for another. In neural networks with millions of parameters, this issue is pervasive, leading to the classic problems of **diminishing gradients** and **oscillations**.


### Finding the Optimal Learning Rate

For a quadratic, **convex** function, the optimal learning rate $\alpha^*$ balances the convergence of all dimensions by equalizing their convergence rates:

$$|1 - \alpha \lambda_{\min}| = |1 - \alpha \lambda_{\max}|$$

Setting these equal with opposite signs (since one is increasing, the other decreasing):

$$1 - \alpha^* \lambda_{\min} = -(1 - \alpha^* \lambda_{\max})$$

Solving for $\alpha^*$:

$$\alpha^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}$$

**Example:** For our 2D case with $\lambda_{\min}=1$ and $\lambda_{\max}=4$:

$$\alpha^* = \frac{2}{1 + 4} = \frac{2}{5} = 0.4$$

The corresponding **optimal convergence rate** is:

$$r(\alpha^*) = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{4-1}{4+1} = \frac{3}{5}$$

### Implications for Deep Learning

This analysis reveals why gradient descent struggles in deep learning:

1. **Millions of dimensions** make it impossible to choose a single learning rate that works well for all parameters
2. **Ill-conditioned loss landscapes** ($\lambda_{\max} \gg \lambda_{\min}$) lead to slow convergence or oscillations
3. The need for **adaptive learning rates** that adjust per parameter or per dimension

These challenges motivate modern optimizers like **Adam**, **RMSprop**, and **AdaGrad**, which use adaptive, per-parameter learning rates based on gradient history. These methods approximate the benefits of second-order information (curvature) without the computational cost of computing and inverting the Hessian matrix.