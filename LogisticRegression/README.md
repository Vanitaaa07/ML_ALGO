# Logistic Regression from Scratch (with Gradient Descent)

This project implements **Logistic Regression** using **Gradient Descent** without using any machine learning libraries.  
It helps you understand how binary classification problems can be solved using the **sigmoid function** and **gradient-based optimization**.

---

## What is Logistic Regression?

**Logistic Regression** is a supervised machine learning algorithm used for **binary classification** problems, where the output variable has two classes like **pass/fail**, **yes/no**, or **0/1**.

Despite the name **"regression"**, Logistic Regression is used for classification, not regression.

To predict a binary output — 0 or 1.

Unlike **Linear Regression**, which predicts continuous outputs, Logistic Regression predicts **probabilities** by applying the **sigmoid function**, which maps any real-valued number into the range [0, 1].

---

## Example Use Case

**Predicting whether a tumor is benign or malignant** based on features like size, texture, or shape.

---

### Formula:

        ŷ = sigmoid(w·X + b)

Where:
- `X` = input features  
- `w` = weights  
- `b` = bias (intercept)  
- `ŷ` = predicted output (probability)  
- `sigmoid(z)` = \( \frac{1}{1 + \exp(-z)} \)

---

## 🧠 How Does Gradient Descent Work?

**Gradient Descent** is an optimization algorithm used to minimize the **Binary Cross-Entropy Loss** by updating the model’s parameters:

- At each step, it computes how much the weights and bias contribute to the error (gradient),
- Then it moves them slightly in the direction that reduces the error.

```text

dw=(1/m)∑(ŷ - y) * x
db=(1/m)∑(ŷ - y)

```
The update rule for weights and bias is:
```text
w = w - α* dw
b = b - α* db

```
Where:
α = learning rate  
m = number of training examples

---
## 🧾 Code Explanation

### 1. Initialization – `__init__()`

```python
def __init__(self, alpha=1e-3, iters=1000):
    self.alpha = alpha      # Learning rate
    self.iters = iters      # Number of training iterations
    self.w = None           # Weights
    self.b = None           # Bias



```

### 2. Sigmoid Function – `_sigmoid()`

```python
def _sigmoid(self, z):
    return 1 / (1 + np.exp(-z))


```

- Squashes real-valued predictions into the [0, 1] range.

### 3. Gradient Calculation – `_gradient()`

```python
def _gradient(self, X, y, y_pred):
    err = y_pred - y
    dw = (1 / self.m) * np.dot(err, X)
    db = (1 / self.m) * np.sum(err)
    return dw, db


```

- Calculates gradients of weights and bias with respect to the loss.

### 4. Parameter Update – `_update_param()`

```python
def _update_param(self, dw, db):
    self.w -= self.alpha * dw
    self.b -= self.alpha * db



```

- Updates model parameters using gradient descent.

### 5. Prediction (Probability) – `predict()`

```python
def predict(self, X):
    z = np.dot(X, self.w) + self.b
    return self._sigmoid(z)



```

- Predicts probabilities using the sigmoid function.

### 6. Training – `fit()`

```python
def fit(self, X, y):
    self.m, self.n = X.shape
    self.w = np.zeros(self.n)
    self.b = 0

    for _ in range(self.iters):
        y_pred = self.predict(X)
        dw, db = self._gradient(X, y, y_pred)
        self._update_param(dw, db)

```

- Initializes parameters and updates them using gradient descent for each iteration.


### 7. Final Prediction – `final_predict()`

```python
def final_predict(self, X):
    return self.predict(X)

```

- A wrapper for making predictions on test data.


## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# Sample Data: Hours studied vs Pass (1)/Fail (0)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Initialize and train model
model = LogisticRegression(alpha=0.1, iters=1000)
model.fit(X, y)

# Predict and visualize
X_test = np.linspace(1, 8, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, color='red', label='Training Data')
plt.plot(X_test, y_pred, color='blue', label='Sigmoid Prediction')
plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
plt.title('Logistic Regression: Pass or Fail')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()

# Predict on new values
print("Prediction for 3.5 hours studied:", model.predict(np.array([[3.5]])))
print("Prediction for 6.5 hours studied:", model.predict(np.array([[6.5]])))


```


## Try This:

- Change the learning rate (`alpha`) and number of iterations.

- Try on datasets with multiple features (`use X = np.array([[1, 2], [3, 4], ...])`).

- Add a threshold to classify predictions (e.g., class 1 if probability > 0.5).

## Learn More
- Binary Cross-Entropy Loss

- Decision Boundary

- Feature Scaling

- Multiclass Logistic Regression (Softmax)

- Regularization in Logistic Regression

## Why Build This?

Implementing Logistic Regression from scratch deepens your understanding of how models learn to classify data. It is a fundamental step toward mastering more complex algorithms in machine learning

## Requirements

- Python 3.x

- NumPy

- Matplotlib

You can install NumPy using:

```bash
pip install numpy matplotlib

```

## 
