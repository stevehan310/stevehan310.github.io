---
layout: single
title: "Implementing Lasso Regression with ISTA (NumPy vs PyTorch)"
author: "Steve Han"
tags: [Python, Machine Learning]
categories: ML
---

## Problem Definition

Lasso Regression adds an L1 regularization term to ordinary linear regression:

$$\min_{w, b} \frac{1}{n}\sum_{i=1}^{n}(y_i - w^Tx_i - b)^2 + \alpha \|w\|_1$$

The problem is that the L1 term is not differentiable at $w=0$, so plain gradient descent doesn't directly apply. ISTA solves this by splitting the optimization into two alternating steps:

1. **Gradient step**: apply gradient descent only to the differentiable MSE loss
2. **Proximal step**: apply the proximal operator of the L1 term, i.e. **soft-thresholding**

$$w_{temp} = w - \eta \nabla_w \text{MSE}, \qquad w_{new} = S(w_{temp}, \eta\alpha)$$

where the soft-thresholding operator is defined as:

$$S(w, \lambda) = \text{sign}(w) \cdot \max(|w| - \lambda, 0)$$

Any weight whose absolute value is smaller than $\lambda$ is pushed exactly to zero — this is the mechanism behind Lasso's feature-selection (sparsity) property. The bias term is not regularized, so it only receives the gradient step, never the soft-thresholding step.

## 1. NumPy Implementation

### soft_threshold

```python
import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.

    S(w, λ) = sign(w) * max(|w| - λ, 0)
    """
    # sign(w) preserves direction, max(|w| - threshold, 0) shrinks magnitude
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
```

Without `np.maximum(..., 0)`, a value that was originally positive could flip to negative after subtraction, so clamping against zero is essential.

### ISTA Main Loop

```python
def l1_regularization_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.1,
    learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4
) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for iteration in range(max_iter):
        # Predict with current weights and bias
        y_pred = X @ weights + bias
        error = y_pred - y  # residual

        # Compute gradients of MSE loss
        grad_w = (2 / n_samples) * (X.T @ error)
        grad_b = (2 / n_samples) * np.sum(error)

        # Step 1: gradient descent step
        w_temp = weights - learning_rate * grad_w
        bias_new = bias - learning_rate * grad_b  # bias is not regularized

        # Step 2: proximal step (soft-thresholding) applied only to weights
        weights_new = soft_threshold(w_temp, learning_rate * alpha)

        # Check convergence: L2 distance between old and new weight vectors
        weight_change = np.linalg.norm(weights_new - weights)

        weights = weights_new
        bias = bias_new

        if weight_change < tol:
            break

    return weights, bias
```

### A Note on the Convergence Check

```python
weight_change = np.linalg.norm(weights_new - weights)
```

`np.linalg.norm(v)` computes the distance of vector `v` from the origin (its magnitude). By first taking the **difference vector** `weights_new - weights` and then measuring its norm, we effectively compute the **Euclidean distance between the two weight vectors**:

$$\|w_{new} - w_{old}\|_2 = \sqrt{\sum_i (w_{new,i} - w_{old,i})^2}$$

Once this value drops below `tol`, we assume the weights have essentially stopped moving and break out of the loop early. This computation must happen **before** `weights` is overwritten with `weights_new` — otherwise the difference is always zero and the loop would break on the very first iteration.

## 2. PyTorch Implementation — Manual Gradient Version

The logic stays identical to the NumPy version; only the operators change.

```python
import torch

def soft_threshold(w: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(w) * torch.clamp(torch.abs(w) - threshold, min=0)


def l1_regularization_gradient_descent(
    X: torch.Tensor, y: torch.Tensor, alpha: float = 0.1,
    learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4
) -> tuple:
    n_samples, n_features = X.shape
    weights = torch.zeros(n_features, dtype=torch.float32)
    bias = torch.tensor(0.0, dtype=torch.float32)

    # Gradients are computed manually (closed-form), so no autograd tracking needed
    with torch.no_grad():
        for iteration in range(max_iter):
            y_pred = X @ weights + bias
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error)
            grad_b = (2 / n_samples) * torch.sum(error)

            w_temp = weights - learning_rate * grad_w
            bias_new = bias - learning_rate * grad_b

            weights_new = soft_threshold(w_temp, learning_rate * alpha)

            weight_change = torch.norm(weights_new - weights)

            weights = weights_new
            bias = bias_new

            if weight_change < tol:
                break

    return weights, bias
```

Key mapping between the two libraries:

| NumPy | PyTorch |
|---|---|
| `np.sign` | `torch.sign` |
| `np.maximum(x, 0)` | `torch.clamp(x, min=0)` |
| `np.linalg.norm` | `torch.norm` |

`with torch.no_grad():` is used here because all gradients are computed manually via closed-form formulas, so autograd tracking is unnecessary. Without it, an unneeded computation graph would build up on every iteration.

## 3. PyTorch Implementation — Autograd Version

This time, instead of deriving the MSE gradient formula by hand, I let `loss.backward()` compute it automatically.

```python
def l1_regularization_gradient_descent(
    X: torch.Tensor, y: torch.Tensor, alpha: float = 0.1,
    learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4
) -> tuple:
    n_samples, n_features = X.shape

    # Leaf tensors that autograd will track gradients for
    weights = torch.zeros(n_features, dtype=torch.float32, requires_grad=True)
    # Scalar (0-dim) tensor for bias, matching the NumPy version's bias = 0.0
    bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    for iteration in range(max_iter):
        # Forward pass: prediction and MSE loss
        y_pred = X @ weights + bias
        loss = torch.mean((y_pred - y) ** 2)

        # Backward pass: autograd computes gradients w.r.t. weights and bias
        loss.backward()

        # Proximal step must NOT be tracked by autograd
        # (soft-thresholding is not part of the differentiable loss graph)
        with torch.no_grad():
            w_temp = weights - learning_rate * weights.grad
            bias_new = bias - learning_rate * bias.grad

            weights_new = soft_threshold(w_temp, learning_rate * alpha)

            weight_change = torch.norm(weights_new - weights)

            # In-place update so 'weights' remains the same leaf tensor
            weights.copy_(weights_new)
            bias.copy_(bias_new)

        # Gradients accumulate by default — must reset every iteration
        weights.grad.zero_()
        bias.grad.zero_()

        if weight_change < tol:
            break

    return weights.detach(), bias.detach()
```

### New Things to Watch Out for with Autograd

1. **`requires_grad=True`** — tells PyTorch to track operations on this tensor so it can later compute gradients for it.
2. **Wrapping the proximal step in `with torch.no_grad():`** — soft-thresholding is non-differentiable, so it must not be part of the backward graph.
3. **`weights.copy_(weights_new)`** — reassigning with `weights = weights_new` would replace the leaf tensor entirely, breaking `.grad` tracking on the next iteration. `copy_()` updates the values in place while keeping the same tensor object.
4. **`weights.grad.zero_()`** — PyTorch accumulates `.grad` by default. Without resetting it every iteration, gradients grow unboundedly and training diverges. This is reportedly one of the most common bugs when writing a PyTorch training loop by hand.
5. **`.detach()`** — the returned tensors still carry `requires_grad=True`. Detaching cuts off graph tracking so the values can be safely used or converted (e.g. to NumPy) downstream.

## 4. Debugging Log: A Bias Shape Bug

Running the test code, `weights` matched the expected shape, but `bias` didn't.

```python
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32)
y = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

weights, bias = l1_regularization_gradient_descent(X, y, alpha=0.1)
print(f"weights shape: {tuple(weights.shape)}")  # (3,)  — OK
print(f"bias shape: {bias.shape}")
```

| | Expected | Actual |
|---|---|---|
| bias shape | `torch.Size([])` | `torch.Size([1])` |

The cause was how `bias` was initialized:

```python
bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)  # shape: (1,)
```

`torch.zeros(1, ...)` creates a **1-dimensional vector with one element** — not a scalar. What I actually needed was a **0-dimensional scalar tensor**:

```python
bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # shape: ()
```

Passing a Python scalar (`0.0`) to `torch.tensor()` produces a 0-dim tensor. The `1` in `torch.zeros(1)` is interpreted as part of the shape tuple, not as "the value 1" — an easy thing to overlook.

Fortunately, fixing that single line was enough. Everything downstream (broadcasting in `X @ weights + bias`, `bias.grad.zero_()`, `bias.copy_(bias_new)`) worked correctly without any further changes, since adding a `()`-shaped scalar to a `(n_samples,)` tensor broadcasts automatically, and `bias.grad` is created with the same shape (`()`) as `bias` itself.

## Takeaways

- ISTA splits optimization into a **differentiable part (MSE)** handled by gradient descent and a **non-differentiable part (L1)** handled by a proximal operator (soft-thresholding).
- Porting from NumPy to PyTorch is mostly a matter of swapping API names, but **tensor shapes** — especially scalar `()` vs. single-element vector `(1,)` — need explicit attention.
- Using autograd removes the need to derive gradient formulas by hand, but introduces PyTorch-specific bookkeeping: **gradient accumulation**, **leaf tensor preservation**, and **no_grad blocks**.
