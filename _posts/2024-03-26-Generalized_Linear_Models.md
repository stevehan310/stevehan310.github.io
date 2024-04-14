---
layout: single
title: "Generalized Linear Models"
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


# Generalized Linear Model


```python
import numpy as np
from numpy.random import uniform, normal, poisson, binomial
from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

# Generate example data
np.random.seed(42)  # Set seed for reproducibility
```

## 1. Linear Regression


```python
# generate simulation data
np.random.seed(5)
n_sample = 100
a = 0.5
b = 1
sd = 0.5

x = uniform(1, 5, size=n_sample)
mu = a * x + b # linear predictor is a * x + b, link function is y=x
y = normal(mu, sd) # Probability distribution is normal distribution
```


```python
slope, intercept, r_value, p_value, std_err  = stats.linregress(x, y)
xvals = np.array([min(x), max(x)])
yvals = slope * xvals + intercept
print(f'actual slope is {a}, estimated slope is {slope:0.4f}')
print(f'actual intercept is {b}, estimated intercept is {intercept:0.4f}')
print(f'actual standard error is {sd}, estimated standard error is {std_err:0.4f}')
print(f'p-value of the model is {p_value:0.4f}')
```

    actual slope is 0.5, estimated slope is 0.4865
    actual intercept is 1, estimated intercept is 1.0601
    actual standard error is 0.5, estimated standard error is 0.0432
    p-value of the model is 0.0000



```python
plt.scatter(x, y, s=10, alpha=0.9, label='data')
plt.plot(xvals, yvals, color='red', label='fitted line')
plt.legend()
```

![Alt text for broken image link](/assets/images/GLM/output_5_1.png){:class="img-responsive"}
    


## 2. Poisson Regression

<b>Three cases when Poisson Regression should be applied:</b>
a. When there is an exponential relationship between x and y
b. When the increase in X leads to an increase in the variance of Y
c. When Y is a discrete variable and must be positive


```python
# generate simulation data
n_sample = 100
a = 0.5
b = 1

x = uniform(1, 5, size=n_sample) 
mu = np.exp(a * x + b) # Linear predictor is a * x + b, 
                       # Link function is log function 
                       # (This is why x and y should have an exponential relationship)
y = poisson(mu) # Probability distribution is Poisson Distribution (This is why Y is a positive discrete variable)  
plt.scatter(x, y,  s=20, alpha=0.8)
```

![test1](/assets/images/GLM/output_8_1.png){:class="img-responsive"}


```python
exog, endog = sm.add_constant(x), y

# Poisson regression
mod = sm.GLM(endog, exog, family=sm.families.Poisson(link=sm.families.links.log()))
res = mod.fit()
display(res.summary())

y_pred = res.predict(exog)

idx = x.argsort()
x_ord, y_pred_ord = x[idx], y_pred[idx]
plt.plot(x_ord, y_pred_ord, color='red')
plt.scatter(x, y,  s=10, alpha=0.9)
plt.xlabel("X")
plt.ylabel("Y")
```

    /Users/steve.han/miniconda3/lib/python3.11/site-packages/statsmodels/genmod/families/links.py:13: FutureWarning: The log link alias is deprecated. Use Log instead. The log link alias will be removed after the 0.15.0 release.
      warnings.warn(



<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>  <td>   100</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>    98</td> 
</tr>
<tr>
  <th>Model Family:</th>         <td>Poisson</td>     <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Link Function:</th>          <td>log</td>       <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -263.39</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 06 Apr 2024</td> <th>  Deviance:          </th> <td>  94.720</td>
</tr>
<tr>
  <th>Time:</th>                <td>01:20:47</td>     <th>  Pearson chi2:      </th>  <td>  95.3</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>4</td>        <th>  Pseudo R-squ. (CS):</th>  <td>0.9775</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    1.0606</td> <td>    0.096</td> <td>   10.996</td> <td> 0.000</td> <td>    0.872</td> <td>    1.250</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.4778</td> <td>    0.026</td> <td>   18.627</td> <td> 0.000</td> <td>    0.428</td> <td>    0.528</td>
</tr>
</table>





![Alt text for broken image link](/assets/images/GLM/output_9_3.png){:class="img-responsive"}    
    


## 3. Logistic Regression


```python
## logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


xx = np.linspace(-10, 10)
plt.plot(xx, logistic(xx))
```




![Alt text for broken image link](/assets/images/GLM/output_11_1.png){:class="img-responsive"}    
    



```python
n_sample = 100
a = 1.5
b = -4

x = uniform(1, 5, size=n_sample)
x = np.sort(x)

q = logistic(a * x + b) # Linear predictor is a * x + b, 
                        # Link function is logit function 
y = binomial(n=1, p=q) # Probability distribution is binomial distribution (Bernoulli distribution can be other option)
plt.scatter(x, y,  s=10, alpha=0.9)
```




    <matplotlib.collections.PathCollection at 0x138a3ca90>




    
![Alt text for broken image link](/assets/images/GLM/output_12_1.png){:class="img-responsive"}        



```python
exog, endog = sm.add_constant(x), y

# Logistic regression
mod = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.logit()))
res = mod.fit()
display(res.summary())

y_pred = res.predict(exog)

idx = x.argsort()
x_ord, y_pred_ord = x[idx], y_pred[idx]
plt.plot(x_ord, y_pred_ord, color='r')
plt.scatter(x, y,  s=10, alpha=0.9)
plt.xlabel("X")
plt.ylabel("Y")
```

    /Users/steve.han/miniconda3/lib/python3.11/site-packages/statsmodels/genmod/families/links.py:13: FutureWarning: The logit link alias is deprecated. Use Logit instead. The logit link alias will be removed after the 0.15.0 release.
      warnings.warn(



<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>  <td>   100</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>    98</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -42.196</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 06 Apr 2024</td> <th>  Deviance:          </th> <td>  84.392</td>
</tr>
<tr>
  <th>Time:</th>                <td>01:32:33</td>     <th>  Pearson chi2:      </th>  <td>  101.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>5</td>        <th>  Pseudo R-squ. (CS):</th>  <td>0.3896</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -4.6545</td> <td>    0.987</td> <td>   -4.714</td> <td> 0.000</td> <td>   -6.590</td> <td>   -2.719</td>
</tr>
<tr>
  <th>x1</th>    <td>    1.7746</td> <td>    0.340</td> <td>    5.214</td> <td> 0.000</td> <td>    1.108</td> <td>    2.442</td>
</tr>
</table>





    Text(0, 0.5, 'Y')
    
![Alt text for broken image link](/assets/images/GLM/output_13_3.png){:class="img-responsive"}        


## 4. Custom GLM

<b>Let's create a glm model with conditions below</b>
a. The relationship between x and y is an exponential relationship
b. The variance of y is constant when x increases.
c. y can be either discret or continuous variable and also can be negative


```python
# generate simulation data
n_sample = 100
a = 0.5
b = 1
sd = 0.5

x = uniform(-3, 3, size=n_sample) 
mu = np.exp(a * x + b) # Linear predictor is a * x + b, 
                       # Link function is log function 
                       # (This is why x and y should have an exponential relationship)
y = normal(mu, sd) # Probability distribution is Normal Distribution
plt.scatter(x, y,  s=10, alpha=0.9)
```




    <matplotlib.collections.PathCollection at 0x138c3c610>




    
![Alt text for broken image link](/assets/images/GLM/output_16_1.png){:class="img-responsive"}        



```python
exog, endog = sm.add_constant(x), y

mod = sm.GLM(endog, exog, family=sm.families.Gaussian(link=sm.families.links.log()))
res = mod.fit()
display(res.summary())

y_pred = res.predict(exog)

idx = x.argsort()
x_ord, y_pred_ord = x[idx], y_pred[idx]
plt.plot(x_ord, y_pred_ord, color='red')
plt.scatter(x, y,  s=10, alpha=0.9)
plt.xlabel("X")
plt.ylabel("Y")
```

    /Users/steve.han/miniconda3/lib/python3.11/site-packages/statsmodels/genmod/families/links.py:13: FutureWarning: The log link alias is deprecated. Use Log instead. The log link alias will be removed after the 0.15.0 release.
      warnings.warn(



<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>  <td>   100</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>    98</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Gaussian</td>     <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Link Function:</th>          <td>log</td>       <th>  Scale:             </th> <td> 0.27094</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -75.601</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 06 Apr 2024</td> <th>  Deviance:          </th> <td>  26.552</td>
</tr>
<tr>
  <th>Time:</th>                <td>01:44:12</td>     <th>  Pearson chi2:      </th>  <td>  26.6</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>  Pseudo R-squ. (CS):</th>  <td> 1.000</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    0.9964</td> <td>    0.025</td> <td>   40.561</td> <td> 0.000</td> <td>    0.948</td> <td>    1.045</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.4940</td> <td>    0.011</td> <td>   46.129</td> <td> 0.000</td> <td>    0.473</td> <td>    0.515</td>
</tr>
</table>





    Text(0, 0.5, 'Y')

    
![Alt text for broken image link](/assets/images/GLM/output_17_3.png){:class="img-responsive"}        

