---
layout: single
title: "Implementing Gradient Descent From Scratch: NumPy vs PyTorch"
author: "Steve Han"
tags: [Python, Machine Learning]
categories: ML

---

## Problem 1: Linear Regression with Gradient Descent

### The Problem

```python
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations

    Returns:
        Learned weights as a 1D array of shape (n,)
    """
```

`X` is an `(m, n)` feature matrix where `m` is the number of samples and `n` is the number of features (the first column is all `1`s, which lets a single weight vector also learn the intercept). `y` is the target vector, `alpha` is the learning rate, and `iterations` is how many gradient descent steps to run.

### The Math

This problem uses the standard textbook cost function with a `1/2` factor baked in:

$$L(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

The `1/2` is there for a reason: when you differentiate a square, the exponent `2` pops out front by the chain rule. Pre-multiplying by `1/2` makes that `2` cancel out cleanly:

$$\frac{\partial L}{\partial \theta} = \frac{1}{m}X^T(X\theta - y)$$

So the gradient descent update rule is:

```
gradient = (1/m) * X.T @ error
theta = theta - alpha * gradient
```

### NumPy Implementation (Manual Gradient)

```python
import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    y = y.reshape(-1, 1)          # Ensure y is a column vector (m, 1)
    theta = np.zeros((n, 1))      # Initialize weights to zeros

    for _ in range(iterations):
        y_pred = X @ theta                  # Predicted values, shape (m, 1)
        error = y_pred - y                  # Prediction error
        gradient = (1 / m) * (X.T @ error)  # Gradient of cost w.r.t. theta
        theta = theta - alpha * gradient    # Update rule

    return theta.flatten()
```

**Why reshape `y` and `theta` into column vectors?** NumPy treats a `(m,)` 1D array and a `(m, 1)` 2D array differently under broadcasting. If `y_pred` (shape `(m, 1)`) is subtracted from a 1D `y` (shape `(m,)`), broadcasting can silently produce an unintended `(m, m)` matrix instead of an elementwise `(m, 1)` difference. Reshaping avoids this class of bug entirely.

### PyTorch Implementation — Two Ways

#### Version A: Manual Gradient (mirrors the NumPy code)

```python
import torch

def linear_regression_gradient_descent_torch(X: torch.Tensor, y: torch.Tensor, alpha: float, iterations: int) -> torch.Tensor:
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = torch.zeros((n, 1), dtype=X.dtype)

    for _ in range(iterations):
        y_pred = X @ theta
        error = y_pred - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient

    return theta.flatten()
```

This is a literal translation of the NumPy math into PyTorch tensor operations — no autograd involved.

#### Version B: PyTorch Autograd

```python
import torch

def linear_regression_gradient_descent_autograd(X: torch.Tensor, y: torch.Tensor, alpha: float, iterations: int) -> torch.Tensor:
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = torch.zeros((n, 1), dtype=X.dtype, requires_grad=True)  # Track gradients

    for _ in range(iterations):
        y_pred = X @ theta
        loss = torch.mean((y_pred - y) ** 2) / 2   # 1/(2m) * sum(error^2), matches manual formula

        loss.backward()  # Autograd computes dLoss/dtheta automatically

        with torch.no_grad():          # Don't track this update step
            theta -= alpha * theta.grad
            theta.grad.zero_()         # Reset gradient before next iteration

    return theta.detach().flatten()    # Detach from computation graph before returning
```

Instead of deriving `gradient = (1/m) * X.T @ error` by hand, we just define the cost function (`loss`) and let `loss.backward()` compute the derivative automatically. Note the `/ 2` at the end of the loss — this makes the cost function match `1/(2m) * sum(error^2)` exactly, so autograd produces the same `1/m` gradient we derived manually.

**Three details that matter here:**

1. **`requires_grad=True`** — tells PyTorch to track operations on `theta` so it can be differentiated later.
2. **`with torch.no_grad():`** — the parameter *update* step itself shouldn't be tracked by autograd; without this you'd get errors or an unnecessarily growing computation graph.
3. **`theta.grad.zero_()`** — PyTorch **accumulates** gradients by default. Forgetting to zero them out means each iteration's gradient gets added on top of the previous one, silently corrupting the results.

### A Debugging Detour: `RuntimeError` and `grad_fn`

Early on, comparing output against an expected tensor produced:

```
expected: tensor([1., 2.])
got:      tensor([1., 2.], grad_fn=<DivBackward0>)
```

The values were numerically correct, but the returned tensor still carried its autograd history (`grad_fn`). The fix is to call **`.detach()`** before returning:

```python
return theta.detach().flatten()
```

`.detach()` strips the computation-graph "history" off a tensor and leaves just the plain values — appropriate once training is done and you only need the final numbers.

---

## Problem 2: Batch / Stochastic / Mini-Batch Gradient Descent

### The Problem

```python
def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights
    """
```

This problem generalizes Problem 1 into three variants:

| Method | Samples per update | Updates per epoch |
|---|---|---|
| **Batch** | all `m` samples | 1 |
| **Stochastic (SGD)** | 1 sample | `m` |
| **Mini-batch** | `batch_size` samples | `m / batch_size` |

An **epoch** is one full pass through the dataset — how many *weight updates* happen within that pass depends on the method.

### The Math — A Different Convention

This problem defines its cost function **without** the `1/2` factor:

$$L(\theta) = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

Differentiating this, the `2` from the power rule has nothing to cancel against, so it survives into the gradient:

$$\frac{\partial L}{\partial \theta} = \frac{2}{m}X^T(X\theta - y)$$

This is an important lesson: **the `1/2` factor is a convention, not a law of nature.** Different textbooks/problems define MSE differently, and the "correct" gradient coefficient depends entirely on how the cost function was defined. When in doubt, the fastest way to find the right coefficient is to test candidate formulas against a known expected output.

For reference, here's how that coefficient was verified empirically against `expected = [1.14905239, 0.56176776]`:

| Candidate coefficient | Result |
|---|---|
| `1/m` | `[1.17298353, 0.49076373]` ❌ |
| `1/(2m)` | `[1.1697732, 0.44546986]` ❌ |
| **`2/m`** | **`[1.14905239, 0.56176776]`** ✅ |

### NumPy Implementation (Manual Gradient, All Three Methods)

```python
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m, n = X.shape

    if method == 'batch':
        for epoch in range(n_epochs):
            y_pred = X @ weights
            error = y_pred - y
            gradient = (2 / m) * (X.T @ error)
            weights = weights - learning_rate * gradient
        return weights.flatten()

    elif method == 'stochastic':
        for epoch in range(n_epochs):
            for i in range(m):
                Xi = X[i:i+1]
                yi = y[i:i+1]
                y_pred = Xi @ weights
                error = y_pred - yi
                gradient = 2 * (Xi.T @ error)     # no averaging — single sample
                weights = weights - learning_rate * gradient
        return weights.flatten()

    elif method == 'mini_batch':
        for epoch in range(n_epochs):
            for i in range(0, m, batch_size):
                Xi = X[i:(i + batch_size)]
                yi = y[i:(i + batch_size)]
                y_pred = Xi @ weights
                error = y_pred - yi
                gradient = (2 / batch_size) * (Xi.T @ error)
                weights = weights - learning_rate * gradient
        return weights.flatten()
```

**Key implementation detail — slicing:** the mini-batch loop must slice `X[i : i + batch_size]`, *not* `X[i : i + 1]`. The latter is a classic off-by-one-style bug: it always grabs exactly one sample regardless of `batch_size`, silently turning "mini-batch" into "stochastic" while also skipping most of the dataset.

**Stochastic gradient's coefficient (`2`, no `1/m`)** makes sense because, for a single sample, there's no averaging to do — the "batch" is size 1, so the `1/m` in the batch formula naturally becomes `1/1 = 1`, leaving just the `2` from the power rule.

### PyTorch Implementation — Manual Gradient

```python
import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor,
                    learning_rate: float, n_epochs: int,
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32).clone()
    m, n = X.shape

    if method == 'batch':
        for epoch in range(n_epochs):
            y_pred = X @ weights
            error = y_pred - y
            gradient = (2 / m) * (X.T @ error)
            weights = weights - learning_rate * gradient

    elif method == 'stochastic':
        for epoch in range(n_epochs):
            for i in range(m):
                Xi = X[i:i+1]
                yi = y[i:i+1]
                y_pred = Xi @ weights
                error = y_pred - yi
                gradient = 2 * (Xi.T @ error)
                weights = weights - learning_rate * gradient

    elif method == 'mini_batch':
        for epoch in range(n_epochs):
            for i in range(0, m, batch_size):
                Xi = X[i:(i + batch_size)]
                yi = y[i:(i + batch_size)]
                y_pred = Xi @ weights
                error = y_pred - yi
                gradient = (2 / batch_size) * (Xi.T @ error)
                weights = weights - learning_rate * gradient

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights
```

This is a near 1:1 translation of the NumPy version — `@` and `.T` work identically in PyTorch. Two additions matter for correctness:

- **`torch.as_tensor(X, dtype=torch.float32)`**: even though the type hint says `X: torch.Tensor`, Python does **not enforce type hints at runtime**. If `X` arrives as an `int64` tensor (very easy to get by accident — `torch.tensor([[1,1],[2,1]])` defaults to `int64`), matrix multiplication with a `float32` weight tensor throws a dtype mismatch error. `torch.as_tensor(..., dtype=torch.float32)` is a defensive one-liner that normalizes dtype regardless of what came in, and is a no-op (no copy) if the tensor already matches.
- **`.clone()`**: `torch.as_tensor()` does **not** copy data when the input is already a tensor of the target dtype — it returns an alias pointing at the same memory. In the code above, weight updates are done via *reassignment* (`weights = weights - ...`), which never mutates the original memory, so `.clone()` isn't strictly required here. But it's cheap insurance: if the update logic is ever refactored to use an in-place operator (`weights -= ...`), an alias would silently corrupt the caller's original tensor. This is the exact same category of bug that `df.copy()` guards against in pandas — both guard against unintended mutation through shared references.

### PyTorch Implementation — Autograd

The manual version above never actually uses PyTorch's signature feature: automatic differentiation. Here's the same three methods, but letting `loss.backward()` compute every gradient instead of deriving `2/m`, `2`, and `2/batch_size` by hand:

```python
import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor,
                    learning_rate: float, n_epochs: int,
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32).clone().requires_grad_(True)
    m, n = X.shape

    if method == 'batch':
        for epoch in range(n_epochs):
            y_pred = X @ weights
            loss = torch.mean((y_pred - y) ** 2)   # MSE over the full dataset

            loss.backward()                         # autograd computes dLoss/dweights

            with torch.no_grad():
                weights -= learning_rate * weights.grad
                weights.grad.zero_()

    elif method == 'stochastic':
        for epoch in range(n_epochs):
            for i in range(m):
                Xi = X[i:i+1]
                yi = y[i:i+1]
                y_pred = Xi @ weights
                loss = torch.mean((y_pred - yi) ** 2)  # MSE over a single sample

                loss.backward()

                with torch.no_grad():
                    weights -= learning_rate * weights.grad
                    weights.grad.zero_()

    elif method == 'mini_batch':
        for epoch in range(n_epochs):
            for i in range(0, m, batch_size):
                Xi = X[i:(i + batch_size)]
                yi = y[i:(i + batch_size)]
                y_pred = Xi @ weights
                loss = torch.mean((y_pred - yi) ** 2)  # MSE over the mini-batch

                loss.backward()

                with torch.no_grad():
                    weights -= learning_rate * weights.grad
                    weights.grad.zero_()

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights.detach()
```

Notice we never write `2/m`, `2`, or `2/batch_size` anywhere — `torch.mean((y_pred - y) ** 2)` defines the cost, and `loss.backward()` derives the correct coefficient automatically for every method. This is the real value of autograd: it removes an entire category of bugs (getting the derivative's constant factor wrong).

Verified numerically against the manual-gradient version, all three methods match to floating-point precision:

```
batch:      manual [1.17298353, 0.49076373] with 1/m  → wrong coefficient
            manual [1.14905239, 0.56176776] with 2/m  → ✅
            autograd                        tensor([1.1491, 0.5618]) → ✅ matches
stochastic: manual [1.0507814,  0.83659454] → autograd [1.0508, 0.8366] ✅ matches
mini_batch: manual [1.10334065, 0.68329431] → autograd [1.1033, 0.6833] ✅ matches
```

### A Second Debugging Detour: `RuntimeError: element 0 of tensors does not require grad`

Switching to autograd introduced a new class of bug. Running:

```python
weights = torch.as_tensor(weights, dtype=torch.float32).clone()   # missing .requires_grad_(True)
...
loss.backward()
```

throws:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The cause: `weights` started life as `torch.zeros(...)`, which defaults to `requires_grad=False`. Neither `torch.as_tensor()` nor `.clone()` changes that flag — if the input didn't require grad, the output doesn't either. Since `weights.requires_grad` was `False`, every downstream tensor (`y_pred`, `loss`) also had `requires_grad=False` and no `grad_fn`, so there was nothing for `.backward()` to differentiate through.

**Fix:** explicitly opt in with `.requires_grad_(True)`:

```python
weights = torch.as_tensor(weights, dtype=torch.float32).clone().requires_grad_(True)
```

Two smaller bugs surfaced alongside this one and are worth flagging since they're easy to make:

- **`weights.grad.zero()` vs `weights.grad.zero_()`** — PyTorch's in-place methods use a trailing underscore by convention. `zero()` isn't a valid method; it has to be `zero_()`.
- **`loss = 2 * torch.mean((y_pred - y) ** 2)`** — manually multiplying the autograd-computed loss by an extra `2` reintroduces the exact class of bug autograd is supposed to eliminate: it silently doubles the gradient, and if the multiplier isn't applied consistently across all three methods (as happened here — it was missing from the `mini_batch` branch), the three methods end up using different effective learning rates.

---

## Takeaways

1. **The `1/2` factor in MSE is a convention, not a rule.** Always check what the *expected output* implies about which coefficient a specific problem wants, rather than assuming.
2. **NumPy and PyTorch manual-gradient code translate almost 1:1** — the main differences are dtype handling (`torch.as_tensor`) and defensive copying (`.clone()`), not the math itself.
3. **Autograd's value is removing manual derivative bugs**, but it introduces its own failure mode: forgetting `requires_grad=True` on the parameter you intend to differentiate, or forgetting `.grad.zero_()` between iterations (gradients accumulate by default).
4. **Off-by-one slicing bugs are sneaky.** `X[i:i+1]` instead of `X[i:i+batch_size]` still runs without error — it just silently changes what algorithm you're actually implementing.
5. **`.detach()` before returning** any tensor that was part of a training loop's computation graph, or downstream code that compares it to a plain tensor will see a mismatch caused by the lingering `grad_fn`.
