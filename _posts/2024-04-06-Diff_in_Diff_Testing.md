---
layout: single
title: "Difference in Differences"
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

# Difference in Differences in Python

```python
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format

%precision %.2f
```




    '%.2f'




```python
df = pd.read_csv('./data/employment.csv')
```

The dataset is adapted from the dataset in Card and Krueger (1994), which estimates the causal effect of an increase in the state minimum wage on the employment.

On April 1, 1992, New Jersey raised the state minimum wage from 4.25 USD to 5.05 USD while the minimum wage in Pennsylvania stays the same at 4.25 USD. data about employment in fast-food restaurants in NJ (0) and PA (1) were collected in February 1992 and in November 1992. 384 restaurants in total after removing null values The calculation of DID is simple:


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 384 entries, 0 to 383
    Data columns (total 3 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   state          384 non-null    int64  
     1   total_emp_feb  384 non-null    float64
     2   total_emp_nov  384 non-null    float64
    dtypes: float64(2), int64(1)
    memory usage: 9.1 KB



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>total_emp_feb</th>
      <th>total_emp_nov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>40.50</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>13.75</td>
      <td>11.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>8.50</td>
      <td>10.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>34.00</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>24.00</td>
      <td>35.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.state.value_counts()
```




    1    309
    0     75
    Name: state, dtype: int64



# New Jersey data


```python
# New Jersey
df_nj = df[df.state == 0]
df_nj['delta'] = df_nj['total_emp_nov'] - df_nj['total_emp_feb']
print(f'the change of employment from feb to nov in New Jersey is {df_nj.delta.mean():.2f}')
```

    the change of employment from feb to nov in New Jersey is -2.28


# Pennsylvania


```python
df_pa = df[df.state == 1]
df_pa['delta'] = df_pa['total_emp_nov'] - df_pa['total_emp_feb']
print(f'the change of employment from feb to nov in Pennsylvania is {df_pa.delta.mean():.2f}')
```

    the change of employment from feb to nov in Pennsylvania is 0.47


The difference between New Jersey and Pennsylvania is 0.47 - (-2.28) = 2.75. However, how can we know that this is statistically significant? We use the linear regression. 

𝑦 = 𝛽0 + 𝛽1∗state + 𝛽2∗month + 𝛽3∗(interaction) + 𝜀
 
state is 0 for the control group (New Jersey) and 1 for the treatment group (Pennsylvania)
month is 0 for before (Feb) and 1 for after (Nov)
we can insert the values of state and month using the table below and see that coefficient (𝛽3) of the interaction of state and month is the value for DID


```python
df_train = pd.melt(df, id_vars=['state'], value_vars=['total_emp_feb', 'total_emp_nov'], var_name='month', value_name='total_emp')
```


```python
df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>month</th>
      <th>total_emp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>total_emp_feb</td>
      <td>40.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>total_emp_feb</td>
      <td>13.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>total_emp_feb</td>
      <td>8.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>total_emp_feb</td>
      <td>34.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>total_emp_feb</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>1</td>
      <td>total_emp_nov</td>
      <td>23.75</td>
    </tr>
    <tr>
      <th>764</th>
      <td>1</td>
      <td>total_emp_nov</td>
      <td>17.50</td>
    </tr>
    <tr>
      <th>765</th>
      <td>1</td>
      <td>total_emp_nov</td>
      <td>20.50</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>total_emp_nov</td>
      <td>20.50</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>total_emp_nov</td>
      <td>25.00</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 3 columns</p>
</div>




```python
df_train.loc[df_train.month == 'total_emp_feb', 'month'] = 0
df_train.loc[df_train.month == 'total_emp_nov', 'month'] = 1
df_train['interaction'] = df_train['state'] * df_train['month']
```


```python
df_train.state.value_counts()
```




    1    618
    0    150
    Name: state, dtype: int64




```python
df_train.month.value_counts()
```




    0    384
    1    384
    Name: month, dtype: int64




```python
df_train.month.value_counts()
```




    0    384
    1    384
    Name: month, dtype: int64




```python
df_train.interaction.value_counts()
```




    0    459
    1    309
    Name: interaction, dtype: int64




```python
import statsmodels.formula.api as sm
model = sm.ols('total_emp ~ state + month + interaction', data=df_train).fit()

print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:              total_emp   R-squared:                       0.008
    Model:                            OLS   Adj. R-squared:                  0.004
    Method:                 Least Squares   F-statistic:                     1.947
    Date:                Thu, 25 Apr 2024   Prob (F-statistic):              0.121
    Time:                        21:22:02   Log-Likelihood:                -2817.6
    No. Observations:                 768   AIC:                             5643.
    Df Residuals:                     764   BIC:                             5662.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept           23.3800      1.098     21.288      0.000      21.224      25.536
    month[T.1]          -2.2833      1.553     -1.470      0.142      -5.332       0.766
    interaction[T.1]     2.7500      1.731      1.588      0.113      -0.649       6.149
    state               -2.9494      1.224     -2.409      0.016      -5.353      -0.546
    ==============================================================================
    Omnibus:                      212.243   Durbin-Watson:                   1.835
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              761.734
    Skew:                           1.278   Prob(JB):                    3.90e-166
    Kurtosis:                       7.155   Cond. No.                         11.3
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The p-value for interaction is not statistically significant, which means that the average total employees per restaurant increased after the minimal salary raise by 2.75 FTE (full-time equivalent) but the result may be just due to random factors.
