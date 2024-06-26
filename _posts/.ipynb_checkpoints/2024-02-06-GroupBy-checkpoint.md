---
layout: single
title: "Pandas: Groupby Method"
author: "Steve Han"
tags: [Pandas]
categories: Python
# toc: true
# toc_sticky: true
# # toc_label: "목차"
# # toc_icon: "fas fa-utensils"
# author_profile: false
# # sidebar:
# #   nav: "docs"
# search: true
---
# Pandas data wrangling (1) - Groupby

## Loading packages


```python
import pandas as pd
import numpy as np
import os
path = os.getcwd()
print(path)
```

    /Users/steve.han/git/stevehan310.github.io/notebooks


## Loading data file (Titanic Data from Kaggle Competition)


```python
df = pd.read_csv(os.path.join(path, 'data', 'titanic_train.csv'))
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB


### Data Dictionary<br>
Variable	Definition	    Key<br>
survival	Survival	    0 = No, 1 = Yes<br>
pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd<br>
sex	        Sex	<br>
Age	        Age             in years<br>
sibsp	    # of siblings / spouses aboard the Titanic	<br>
parch	    # of parents / children aboard the Titanic	<br>
ticket	    Ticket number	<br>
fare	    Passenger fare	<br>
cabin	    Cabin number	<br>
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton<br>
<br>
### Variable Notes<br>
pclass: A proxy for socio-economic status (SES)<br>
1st = Upper<br>
2nd = Middle<br>
3rd = Lower<br>
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>
sibsp: The dataset defines family relations in this way...<br>
Sibling = brother, sister, stepbrother, stepsister<br>
Spouse = husband, wife (mistresses and fiancés were ignored)<br>
parch: The dataset defines family relations in this way...<br>
Parent = mother, father<br>
Child = daughter, son, stepdaughter, stepson<br>
Some children travelled only with a nanny, therefore parch=0 for them.<br>


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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Pandas Groupby 


```python
# Creating groupby object with embarked and Pclass
df_agg = df.groupby(['Embarked', 'Pclass'], as_index = False)
```


```python
# Average Fare price per Port of Embarkation and Ticket Class
df_agg.agg(fare_avg = ('Fare', np.mean), # Get the average of Fare per group and add them into a column with name 'fare_avg'
           fare_min = ('Fare', np.min), # Get a minimum of Fare per group and add them into a column with name 'fare_min'
           fare_max = ('Fare', np.max), # Get a maximum of Fare per group and add them into a column with name 'fare_max'
           num_passenger = ('PassengerId', 'nunique') # Get a number of passenger per group and add them into a column with name 'num_passenger'
          )

# np.unique
# np.count_nonzero  
# np.sum – Sum of values
# np.mean – Mean of values
# np.median – Arithmetic median of values
# np.min – Minimum
# np.max – Maximum
# np.std – Standard deviation
# np.var – Variance
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
      <th>Embarked</th>
      <th>Pclass</th>
      <th>fare_avg</th>
      <th>fare_min</th>
      <th>fare_max</th>
      <th>num_passenger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>1</td>
      <td>104.718529</td>
      <td>26.5500</td>
      <td>512.3292</td>
      <td>85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>2</td>
      <td>25.358335</td>
      <td>12.0000</td>
      <td>41.5792</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
      <td>11.214083</td>
      <td>4.0125</td>
      <td>22.3583</td>
      <td>66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q</td>
      <td>1</td>
      <td>90.000000</td>
      <td>90.0000</td>
      <td>90.0000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Q</td>
      <td>2</td>
      <td>12.350000</td>
      <td>12.3500</td>
      <td>12.3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Q</td>
      <td>3</td>
      <td>11.183393</td>
      <td>6.7500</td>
      <td>29.1250</td>
      <td>72</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S</td>
      <td>1</td>
      <td>70.364862</td>
      <td>0.0000</td>
      <td>263.0000</td>
      <td>127</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S</td>
      <td>2</td>
      <td>20.327439</td>
      <td>0.0000</td>
      <td>73.5000</td>
      <td>164</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S</td>
      <td>3</td>
      <td>14.644083</td>
      <td>0.0000</td>
      <td>69.5500</td>
      <td>353</td>
    </tr>
  </tbody>
</table>
</div>




```python
# You can also use 'apply' aggregation with your own function by using lambda
df_agg.apply(lambda x: x.Fare.max() - x.Fare.min()) # Range of Fare from min to max
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
      <th>Embarked</th>
      <th>Pclass</th>
      <th>None</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>1</td>
      <td>485.7792</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>2</td>
      <td>29.5792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
      <td>18.3458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q</td>
      <td>1</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Q</td>
      <td>2</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Q</td>
      <td>3</td>
      <td>22.3750</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S</td>
      <td>1</td>
      <td>263.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S</td>
      <td>2</td>
      <td>73.5000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S</td>
      <td>3</td>
      <td>69.5500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Same apply aggregation with different format
df_agg.agg({'Fare': lambda x: x.max() - x.min()})
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
      <th>Embarked</th>
      <th>Pclass</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>1</td>
      <td>485.7792</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>2</td>
      <td>29.5792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
      <td>18.3458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q</td>
      <td>1</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Q</td>
      <td>2</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Q</td>
      <td>3</td>
      <td>22.3750</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S</td>
      <td>1</td>
      <td>263.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S</td>
      <td>2</td>
      <td>73.5000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S</td>
      <td>3</td>
      <td>69.5500</td>
    </tr>
  </tbody>
</table>
</div>


