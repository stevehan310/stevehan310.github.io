---
layout: single
title: "Python Tips"
author: "Steve Han"
tags: [Python]
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
# 1. Python string type

## 1.a Python Escape Characters


```python
# \n : to break lines within a string
# \t : to add tab spacing between strings
# \\ : to represent the backslash character () as it is
# \' : to represent the single quote character (') as it is
# \" : to represent the double quote character (") as it is
# \r : Carriage return (line break character, moves the cursor to the beginning of the current line)
# \f : Form feed (line break character, moves the cursor to the next line)
# \a : Bell sound (when printed, emits a 'beep' sound from the PC speaker)
# \b : Backspace
# \000 : Null character
```

## 1.b F-string formatting in python


```python
# f-string was introduced in Python 3.6, so it is invalid in Python versions lower than 3.6.
```


```python
y = 3.42134234
f'{y:0.4f}' # up to 4 decimal places
```




    '3.4213'




```python
f'{y:10.4f}' # up to 4 decimal places and set the total number of digits to 10.
```




    '    3.4213'



# 2. Dictionary type


```python
a = {'name':'pey', 'phone':'010-9999-1234', 'birth': '1118'}
print(a.get('nokey')) # Return 'None'
```

    None



```python
print(a['nokey’]) # Return error
```


      Cell In[8], line 1
        print(a['nokey’]) # Return error
                ^
    SyntaxError: unterminated string literal (detected at line 1)




```python
a.get('nokey', 'foo') # When the key is in the dictionary, return the default value 'foo'
```




    'foo'



# 3. 'for' and 'continue' 


```python
marks = [90, 25, 67, 45, 80]
number = 0 
for mark in marks: 
    number = number +1 
    if mark < 60:
        continue 
    print("student #%d passed the exam. " % number)
```

    student #1 passed the exam. 
    student #3 passed the exam. 
    student #5 passed the exam. 


# 4. list comprehension


```python
a = [1,2,3,4]
result = []
for num in a:
    result.append(num*3)

print(result)
```

    [3, 6, 9, 12]



```python
result = [num * 3 for num in a]
print(result)
```

    [3, 6, 9, 12]



```python
result = [num * 3 for num in a if num % 2 == 0] # If you want to multiply 3 to even numbers only
print(result)
```

    [6, 12]


# 5. Functions

## 5.1 *args


```python
def add_many(*args): # '*' converts arguments into tuple format
    result = 0 
    for i in args: 
        result = result + i   # add all numbers from *args
    return result

result = add_many(1, 2, 3)
print(result)
result = add_many(1, 2, 3, 4, 5)
print(result)
```

    6
    15


## 5.b Keyword Arguments **kwargs


```python
def print_kwargs(**kwargs): # '**' converts arguments into dictionary format
    print(kwargs) 
```


```python
print_kwargs(a=1)
```

    {'a': 1}



```python
print_kwargs(name='foo', age=3)
```

    {'name': 'foo', 'age': 3}


# 6. Print


```python
for i in range(5):
    print(i)
```

    0
    1
    2
    3
    4



```python
for i in range(5):
    print(i, end=' ') # Default 'end' is '\n'
```

    0 1 2 3 4 

# 7. Python Decorator


```python
import time

def elapsed(original_func):   # Receives the original function as an argument.
    def wrapper():
        start = time.time()
        result = original_func()    # Executes the original function
        end = time.time()
        print("Function execution time: %f seconds" % (end - start))  # Prints the execution time of the original function.
        return result  # Returns the result of the original function.
    return wrapper

def myfunc():
    print("Function is being executed.")

decorated_myfunc = elapsed(myfunc)
decorated_myfunc()
```

    Function is being executed.
    Function execution time: 0.000403 seconds
