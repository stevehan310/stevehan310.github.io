---
layout: single
title: "Simple Linear Regression Model in PyTorch and Tensorflow"
author: "Steve Han"
tags: [Python]
categories: NN
# toc: true
# toc_sticky: true
# # toc_label: "목차"
# # toc_icon: "fas fa-utensils"
# author_profile: false
# # sidebar:
# #   nav: "docs"
# search: true
---

# Simple Linear Regression Model in PyTorch and Tensorflow


```python
from numpy.random import uniform, normal
import numpy as np
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

np.random.seed(0)  # Set seed for reproducibility

%precision %.4f
```




    '%.4f'




```python
# generate simulation data
n_sample = 100
a = 0.5
b = 10
sd = 10

X = uniform(0, 100, size=n_sample).astype(np.float32)
mu = a * X + b # linear predictor is a * x + b, link function is y=x
Y = normal(mu, sd) # Probability distribution is normal distribution
```


```python
plt.scatter(X, Y, s=10, alpha=0.9, label='data')
#plt.ylim(0,80)
plt.legend()
```




    <matplotlib.legend.Legend at 0x142c8fb90>




![Alt text for broken image link](/assets/images/SLR_NN/output_3_1.png){:class="img-responsive"}


# 1. Linear Regression Model in Tensorflow


```python
import tensorflow as tf
```

    2024-06-11 23:59:09.968485: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, 
                          input_shape=(1,), 
                          kernel_initializer='zeros', 
                          bias_initializer='zeros'
                         )
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), 
              loss='mean_squared_error')

# Train the model
history = model.fit(X, Y, epochs=10, shuffle=False, verbose=1)
```

    Epoch 1/10


    /Users/steve.han/miniconda3/lib/python3.11/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1161.4839  
    Epoch 2/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 165.0661 
    Epoch 3/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 160.1171 
    Epoch 4/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 160.0082 
    Epoch 5/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 159.9909 
    Epoch 6/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 159.9754 
    Epoch 7/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 159.9600 
    Epoch 8/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 159.9446 
    Epoch 9/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 159.9291 
    Epoch 10/10
    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 159.9137 



```python
# Plot the loss over epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```


![Alt text for broken image link](/assets/images/SLR_NN/output_7_0.png){:class="img-responsive"}



```python
# Visualize the predictions
plt.scatter(X, Y, s=10, alpha=0.9, label='Data')
plt.plot(X, model.predict(X), color='red', label='Fitted Line')
plt.legend()
plt.show()
```

    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step 



![Alt text for broken image link](/assets/images/SLR_NN/output_8_1.png){:class="img-responsive"}


```python
# Print the weights of the model
weights, biases = model.layers[0].get_weights()
print(f"Weights: {weights}, Biases: {biases}")
```

    Weights: [[0.6098]], Biases: [0.0348]


# 2. Linear Regression Model in PyTorch


```python
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)
```




    <torch._C.Generator at 0x1548cf210>




```python
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
```


```python
# Define the model
class LinearRegressionModel(nn.Module):
    
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x)
```


```python
model = LinearRegressionModel()
```


```python
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
```


```python
# Train the model
num_epochs = 20
losses = []

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   
    losses.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

    Epoch [1/20], Loss: 1566.5591
    Epoch [2/20], Loss: 352.8644
    Epoch [3/20], Loss: 171.5089
    Epoch [4/20], Loss: 144.4068
    Epoch [5/20], Loss: 140.3534
    Epoch [6/20], Loss: 139.7440
    Epoch [7/20], Loss: 139.6492
    Epoch [8/20], Loss: 139.6313
    Epoch [9/20], Loss: 139.6249
    Epoch [10/20], Loss: 139.6203
    Epoch [11/20], Loss: 139.6158
    Epoch [12/20], Loss: 139.6115
    Epoch [13/20], Loss: 139.6071
    Epoch [14/20], Loss: 139.6027
    Epoch [15/20], Loss: 139.5983
    Epoch [16/20], Loss: 139.5940
    Epoch [17/20], Loss: 139.5896
    Epoch [18/20], Loss: 139.5852
    Epoch [19/20], Loss: 139.5808
    Epoch [20/20], Loss: 139.5765



```python
# Plot the loss over epochs
plt.plot(losses)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```


![Alt text for broken image link](/assets/images/SLR_NN/output_17_0.png){:class="img-responsive"}



```python
# Visualize the predictions
model.eval()
predicted = model(X_tensor).detach().numpy()

plt.scatter(X, Y, s=10, alpha=0.9, label='Data')
plt.plot(X, predicted, color='red', label='Fitted Line')
plt.legend()
plt.show()
```


![Alt text for broken image link](/assets/images/SLR_NN/output_18_0.png){:class="img-responsive"}



```python
# Print the weights of the model
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
```

    linear.weight tensor([[0.6818]])
    linear.bias tensor([0.0237])


### * To have similar weights and biases for both models, the following rules were applied:
##### a. Both models' weights and biases were initialized to zero.
##### b. Data shuffling was disabled (shuffle=False in TensorFlow).
##### c. The same random seed was used in both PyTorch and TensorFlow (torch.manual_seed(0) and np.random.seed(0)).
