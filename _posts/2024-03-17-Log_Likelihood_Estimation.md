---
layout: single
title: "Log Likelihood Estimation"
author: "Steve Han"
tags: [Statistics, Python]
categories: ML
# toc: true
# toc_sticky: true
# # toc_label: "목차"
# # toc_icon: "fas fa-utensils"
# author_profile: false
# # sidebar:
# #   nav: "docs"
# search: true
---


# Log Likelihood Estimation in Python


```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```


```python
# Define the true mean and standard deviation of the normal distribution.
true_mean = 5
true_std_dev = 2
```

## 1. Normal Distribution Parameter Estimation by Log Likelihood

### 1.1 Sample Size 100


```python
# Generate sample data following a normal distribution.
np.random.seed(42)  # Set seed for reproducibility.
sample_data_100 = np.random.normal(loc=true_mean, scale=true_std_dev, size=100)
```


```python
sample_data_100[0:10]
```




    array([5.99342831, 4.7234714 , 6.29537708, 8.04605971, 4.53169325,
           4.53172609, 8.15842563, 6.53486946, 4.06105123, 6.08512009])




```python
# Define the log likelihood function.
def log_likelihood_normal(params, data):
    mean, std_dev = params
    log_likelihoods = norm.logpdf(data, loc=mean, scale=std_dev)
    return -np.sum(log_likelihoods) # This is log likelihood so needs to add negative '-' to maximize likelihood
```


```python
# Set initial guess for parameters.
initial_guess = [0, 1]
# Set bounds for parameters to ensure reasonable values.
bounds = [(-10, 10), (0.1, 10)]

# Calculate MLE by minimizing the log likelihood.
result = minimize(log_likelihood_normal, x0=initial_guess, args=(sample_data_100,), bounds=bounds)
estimated_mean, estimated_std_dev = result.x

# Print the estimated mean and standard deviation.
print("Estimated mean:", f'{estimated_mean:.2f}')
print("Estimated standard deviation:", f'{estimated_std_dev:.2f}')

# Compare with the true mean and standard deviation.
print("True mean:", true_mean)
print("True standard deviation:", true_std_dev)
```

    Estimated mean: 4.79
    Estimated standard deviation: 1.81
    True mean: 5
    True standard deviation: 2


### Sample Size 1000


```python
sample_data_1000 = np.random.normal(loc=true_mean, scale=true_std_dev, size=1000)
# Set initial guess for parameters.
initial_guess = [0, 1]
# Set bounds for parameters to ensure reasonable values.
bounds = [(-10, 10), (0.1, 10)]

# Calculate MLE by minimizing the log likelihood.
result = minimize(log_likelihood, x0=initial_guess, args=(sample_data_1000,), bounds=bounds)
estimated_mean, estimated_std_dev = result.x

# Print the estimated mean and standard deviation.
print("Estimated mean:", f'{estimated_mean:.2f}')
print("Estimated standard deviation:", f'{estimated_std_dev:.2f}')

# Compare with the true mean and standard deviation.
print("True mean:", true_mean)
print("True standard deviation:", true_std_dev)
```

    Estimated mean: 5.09
    Estimated standard deviation: 1.97
    True mean: 5
    True standard deviation: 2


1000 Samples made more accurate estimations!!

## 2. Poisson Distribution Parameter Estimation by Log Likelihood


```python
from scipy.stats import poisson

# True lambda parameter for Poisson distribution
true_lambda = 3.5

# Generate sample data following a Poisson distribution
sample_data_poisson = np.random.poisson(lam=true_lambda, size=100)

# Log likelihood function for Poisson distribution
def log_likelihood_poisson(params, data):
    lambda_param = params[0]
    log_likelihoods = poisson.logpmf(data, mu=lambda_param)
    return -np.sum(log_likelihoods)

# Initial guess for lambda parameter
initial_guess = [1]

# Calculate MLE by minimizing the negative log likelihood
result = minimize(log_likelihood_poisson, initial_guess, args=(sample_data_poisson,))
estimated_lambda = result.x[0]

# Print the estimated lambda parameter
print("Estimated lambda:", f'{estimated_lambda:.2f}')

# Compare with the true lambda parameter
print("True lambda:", true_lambda)
```

    Estimated lambda: 3.40
    True lambda: 3.5

