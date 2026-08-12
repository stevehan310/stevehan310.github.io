---
layout: single
title: "Mastering the Adam Optimizer: From Concept to NumPy & PyTorch Implementation"
author: "Steve Han"
tags: [Python, Machine Learning]
categories: ML
---

Adam (Adaptive Moment Estimation) is one of the most widely used optimizers in deep learning training. In this post, we'll break down the mathematical concept behind Adam, implement it from scratch in NumPy, and then reproduce the same result using PyTorch's built-in optimizer.

---

## 1. What is Adam?

Adam is a variant of gradient descent that combines ideas from two other optimization algorithms: **Momentum** and **RMSProp**.

Standard gradient descent moves in the direction of the gradient by a fixed step size at every iteration. Adam, on the other hand, computes an **individually adaptive learning rate for each parameter**, automatically adjusting how much each parameter should move.

To do this, it tracks two "moments":

1. **First moment `m`** — an exponential moving average of the gradient (similar to momentum)
2. **Second moment `v`** — an exponential moving average of the squared gradient (similar to RMSProp)

It then applies a **bias correction** step to remove the initialization bias before performing the final parameter update.

Adam is known to combine the strengths of **AdaGrad** (which works well with sparse gradients) and **RMSProp** (which works well in online and non-stationary settings), and it is generally considered fairly robust to the choice of hyperparameters.

---

## 2. The Adam Algorithm

Given parameters $\theta$, objective function $f(\theta)$, and its gradient $\nabla_\theta f(\theta)$:

**Initialize**
- Time step $t = 0$
- Parameters $\theta_0$
- First moment vector $m_0 = 0$
- Second moment vector $v_0 = 0$
- Hyperparameters $\alpha$ (learning rate), $\beta_1$, $\beta_2$, $\epsilon$

**While not converged, repeat:**

1. Increment time step: $t = t + 1$
2. Compute gradient: $g_t = \nabla_\theta f_t(\theta_{t-1})$
3. Update biased first moment estimate: $m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$
4. Update biased second raw moment estimate: $v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2$
5. Compute bias-corrected first moment estimate: $\hat{m}_t = m_t / (1 - \beta_1^t)$
6. Compute bias-corrected second raw moment estimate: $\hat{v}_t = v_t / (1 - \beta_2^t)$
7. Update parameters: $\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

| Hyperparameter | Meaning | Typical default |
|---|---|---|
| `learning_rate` (α) | Step size | 0.001 |
| `beta1` | Decay rate for the first moment | 0.9 |
| `beta2` | Decay rate for the second moment | 0.999 |
| `epsilon` | Constant for numerical stability | 1e-8 |

---

## 3. First and Second Moments: Why Do We Need Them?

Both update equations share the same form — an **exponential moving average (EMA)**:

```
new_value = β · old_value + (1-β) · current_observation
```

The closer `β` is to 1, the longer the average "remembers" the past and the smoother it becomes; the closer `β` is to 0, the more sensitive it is to the current observation.

### First moment `m` — smoothing the direction (momentum)

`m` is a moving average of the raw gradient itself. If the loss surface is narrow and winding, causing the gradient to oscillate from step to step, averaging cancels out the oscillating components and leaves only the consistent directional component. This is analogous to a ball rolling down a valley — inertia keeps it moving in a consistent direction despite small bumps.

### Second moment `v` — adapting the step size per parameter (RMSProp)

`v` is a moving average of the **squared** gradient. Since squaring removes the sign, `v` is always positive and represents "how large the recent gradients have been" (their variance/magnitude). Because the update rule divides by `sqrt(v_hat)`, parameters with historically large gradients get smaller steps, while parameters with historically small gradients get relatively larger steps — automatically.

Using `m` alone smooths the direction but applies the same step size to every parameter. Using `v` alone adapts the step size but leaves the direction sensitive to noise. Combining both gives us a **smooth direction with an adaptively scaled step size**.

---

## 4. Why Bias Correction Is Necessary

Because `m` and `v` are initialized to zero, expanding the recurrence reveals something important:

```
m_t = (1-β1) · Σ_{i=1}^{t} β1^(t-i) · g_i
```

The sum of the weights in this expression is exactly `1 - β1^t`, which is much smaller than 1 when `t` is small. In other words, because the initial value was zero, `m` and `v` are **biased toward zero** early in training.

For example, if the gradient were constantly `g = 10`, the uncorrected `m` would only be `1.0` at `t=1` (one-tenth of the true value), but dividing by `1 - β1^t` corrects it to exactly `10.0` starting from `t=1`.

Since `β2 = 0.999` is much closer to 1 than `β1 = 0.9` (it remembers the past for much longer), this bias persists for much longer. In practice, even after `t = 1000` steps, the uncorrected `v` is still only about 63% of its true value.

Without this correction, the effective learning rate would be artificially small during the first several steps of training, slowing down optimization at the start.

---

## 5. Why Is Epsilon Needed?

In the denominator of the update rule, `sqrt(v_hat) + epsilon`, `epsilon` serves two purposes:

1. **Preventing division by zero**: If a parameter's gradient stays close to zero for a while, `v_hat` also converges to zero. Without `epsilon`, `sqrt(v_hat) = 0` would produce a `NaN`. Once a `NaN` appears, it contaminates every subsequent computation and training collapses entirely.
2. **Numerical stability**: Even when `v_hat` isn't exactly zero but is extremely small, the denominator becomes tiny without `epsilon`, which can cause abnormally large update steps. Adding `epsilon` keeps this from blowing up.

A common default is `1e-8`. Setting it too large can dilute the effect of the adaptive learning rate itself.

---

## 6. Implementing Adam from Scratch in NumPy

This version uses a `grad` function (a gradient computation function derived by hand ahead of time) directly. `f` is accepted only for interface consistency and is never actually called.

```python
import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only -- the objective value `f` is never evaluated.

    x = np.array(x0, dtype=np.float64)  # current parameter values

    # First moment vector (momentum term), same shape as x
    m = np.zeros_like(x)
    # Second moment vector (RMSProp-like term), same shape as x
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        g = grad(x)  # compute gradient at current x

        # Update biased first moment estimate (EMA of gradient)
        m = beta1 * m + (1 - beta1) * g

        # Update biased second raw moment estimate (EMA of squared gradient)
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)

        # Bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)

        # Parameter update step
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x
```

### Test

We validate it on $f(x) = x^2$, whose minimum is at $x=0$.

```python
f = lambda x: np.sum(x ** 2)
grad = lambda x: 2 * x

x0 = np.array([5.0, -3.0])
result = adam_optimizer(f, grad, x0, learning_rate=0.5, num_iterations=1000)
print("Result:", result)        # [-3.10e-24, 1.93e-23]
print("f(result):", f(result))  # 3.81e-46
```

Starting from `[5.0, -3.0]`, after 1000 iterations the optimizer converges essentially to zero, confirming the implementation works correctly.

---

## 7. Implementing Adam with PyTorch

This time, instead of manually writing a `grad` function, we let PyTorch's **autograd** compute the gradient automatically, and use `torch.optim.Adam` to perform the update.

```python
import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    # Clone x0 into a new leaf tensor that requires gradient tracking.
    # .clone() avoids mutating the caller's original tensor.
    # .detach() starts a fresh autograd graph.
    x = x0.clone().detach().requires_grad_(True)

    # Built-in Adam optimizer, matching hyperparameters to the manual implementation.
    optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(beta1, beta2), eps=epsilon)

    for _ in range(num_iterations):
        optimizer.zero_grad()   # clear gradients accumulated from the previous step
        loss = f(x)              # forward pass: evaluate the objective function
        loss.backward()          # backward pass: autograd computes gradients into x.grad
        optimizer.step()         # apply the Adam update rule using x.grad

    return x.detach()
```

### Step-by-step explanation

**① `x0.clone().detach().requires_grad_(True)`**
We copy `x0` (`clone`) to protect the caller's original tensor, and detach it from any existing computation graph (`detach`) so it becomes the starting point (leaf tensor) of a fresh autograd graph. Marking it with `requires_grad_(True)` tells PyTorch to track gradients for this tensor so that a later call to `backward()` fills in its gradient.

**② `torch.optim.Adam([x], lr=..., betas=..., eps=...)`**
The state we manually tracked in the NumPy version (`m`, `v`, `t`) is now encapsulated and managed internally by this optimizer object. The hyperparameters are set to match the NumPy implementation exactly.

**③ The four steps inside the loop**

| Code | Role |
|---|---|
| `optimizer.zero_grad()` | Resets `x.grad` accumulated from the previous step to zero (necessary because PyTorch accumulates gradients by default) |
| `loss = f(x)` | Forward pass: evaluates the objective function. PyTorch automatically records the computation graph during this call |
| `loss.backward()` | Backpropagation: uses the chain rule to automatically compute `∂loss/∂x`, storing the result in `x.grad` |
| `optimizer.step()` | Applies the Adam update rule internally, using `x.grad` |

**④ `return x.detach()`**
Since gradient tracking is no longer needed once the function returns, we detach the tensor from the graph and return only its pure value.

### Test

```python
f = lambda x: torch.sum(x ** 2)
x0 = torch.tensor([5.0, -3.0])
result = adam_optimizer(f, None, x0, learning_rate=0.5, num_iterations=1000)
print("Result:", result)        # tensor([-3.05e-24, 1.92e-23])
print("f(result):", f(result))  # 0.0
```

Just like the NumPy version, this converges essentially to zero starting from `[5.0, -3.0]`.

---

## 8. NumPy vs PyTorch: A Side-by-Side Comparison

| Aspect | NumPy version | PyTorch version |
|---|---|---|
| Gradient computation | Calls `grad(x)` directly (a human-derived derivative function) | Evaluates `f(x)`, then calls `loss.backward()` to let autograd compute it automatically |
| Managing `m`, `v`, `t` state | Managed manually as variables | Managed internally by the `torch.optim.Adam` object |
| Whether `f` is called | Never called | Must be called (needed to build the computation graph) |
| Update formula | Written explicitly by hand | Encapsulated inside the optimizer |

Mathematically, both implementations follow the exact same Adam update rule. The only difference is **who computes the gradient** (a hand-written function vs. automatic differentiation) and **who manages the state** (manual variables vs. an optimizer object). Implementing Adam from scratch in NumPy makes it much clearer what's actually happening behind the scenes when you call `optimizer.step()` in PyTorch.

---

## Reference

- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*.
