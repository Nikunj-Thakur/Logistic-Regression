# 📊 Logistic Regression — From Scratch in Python

> A clean, from-scratch implementation of **Logistic Regression** — a supervised machine learning algorithm for binary classification problems.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Key Functions](#key-functions)
- [Visualisations](#visualisations)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Requirements](#requirements)

---

## 🧠 Overview

**Logistic Regression** is a supervised machine learning algorithm used for **classification problems**, where the goal is to predict a categorical outcome such as:

- ✅ / ❌ Yes / No
- 📧 Spam / Not Spam
- 🏥 Disease / No Disease

Unlike Linear Regression, which predicts a continuous value, Logistic Regression squashes the output through a **sigmoid function** to produce a probability between 0 and 1.

---

## ⚙️ How It Works

### 1. Sigmoid / Logistic Function

The core of Logistic Regression is the **sigmoid function**, which maps any real-valued number to a probability between 0 and 1:

```
g(z) = 1 / (1 + e^(-z))
```

where `z = w · X + b` (the linear combination of weights and inputs).

![Sigmoid Function](images/Sigmoid_Function.png)

---

### 2. Decision Boundary

A threshold (typically **0.5**) is applied to the sigmoid output:
- If `g(z) ≥ 0.5` → predict class **1**
- If `g(z) < 0.5` → predict class **0**

![Decision Boundary](images/Decision_Boundary.png)

---

### 3. Cost Function — Log Loss (Binary Cross-Entropy)

Since MSE produces a **non-convex** surface for logistic regression (making optimisation unreliable), we instead use the **logistic cost function**:

```
J(w, b) = -(1/m) * Σ [ y·log(f) + (1 - y)·log(1 - f) ]
```

**Why not MSE?**

![MSE Non-Convex](images/MSE_NonConvex_Function.png)

MSE with sigmoid leads to a non-convex cost landscape with many local minima — gradient descent cannot reliably find the global minimum.

**Logistic Cost Function (convex ✅):**

![Logistic Cost Function](images/Logistic_Cost_Function.png)

The log-loss function is convex, guaranteeing gradient descent will converge to the global minimum.

Optionally, **L2 Regularisation** can be added to prevent overfitting:

```
J_reg = J + (λ / 2m) * Σ w²
```

---

### 4. Binary Classification Plots

![Binary Plots](images/Binary_Plots.png)

Visualisation of binary-classified data points used to train and evaluate the model.

---

## 📁 Project Structure

```
Logistic-Regression/
│
├── logistic_utility_function.py   # Core math: sigmoid, cost, log-sum-exp
├── logistic_regression_model.py   # Model training & gradient descent
│
├── basic_plot/                    # Basic plotting experiments
│
├── images/                        # Output plots & visualisations
│   ├── Sigmoid_Function.png
│   ├── Decision_Boundary.png
│   ├── Logistic_Cost_Function.png
│   ├── MSE_NonConvex_Function.png
│   └── Binary_Plots.png
│
└── README.md
```

---

## 🔧 Key Functions

### `logistic_utility_function.py`

| Function | Description |
|---|---|
| `sigmoid(z)` | Computes `1 / (1 + exp(-z))` — the logistic activation |
| `log_1pexp(x)` | Numerically stable approximation of `log(1 + exp(x))` to avoid overflow |
| `compute_cost_matrix(X, y, w, b, ...)` | Vectorised Binary Cross-Entropy cost with optional L2 regularisation |

#### `sigmoid(z)`
```python
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
```

#### `log_1pexp(x)` — Numerical Stability
Avoids floating-point overflow for large `x` by switching to a stable approximation:
```python
out[i]  = np.log(1 + np.exp(x[i]))   # for x <= threshold (default 20)
out[ni] = x[ni]                        # for x > threshold  (≈ x itself)
```

#### `compute_cost_matrix(X, y, w, b, logistic, lambda_, safe)`
- Supports both **linear** and **logistic** cost via the `logistic` flag
- `safe=True` uses the numerically stable `log_1pexp` path
- Adds **L2 regularisation** scaled by `lambda_`

---

## 📈 Visualisations

| Plot | Description |
|---|---|
| ![Sigmoid](images/Sigmoid_Function.png) | Sigmoid / logistic function curve |
| ![Decision Boundary](images/Decision_Boundary.png) | Learned decision boundary separating classes |
| ![Logistic Cost](images/Logistic_Cost_Function.png) | Convex log-loss cost surface |
| ![MSE Non-Convex](images/MSE_NonConvex_Function.png) | Non-convex MSE surface — why we don't use it |
| ![Binary Plots](images/Binary_Plots.png) | Binary-labelled training data scatter plot |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Nikunj-Thakur/Logistic-Regression.git
cd Logistic-Regression
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib
```

### 3. Run the Model

```bash
python logistic_regression_model.py
```

---

## 💡 Usage

```python
import numpy as np
from logistic_utility_function import sigmoid, compute_cost_matrix

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 1])
w = np.zeros(X.shape[1])
b = 0.0

# Sigmoid predictions
z = X @ w + b
predictions = sigmoid(z)
print("Predictions:", predictions)

# Logistic cost with L2 regularisation
cost = compute_cost_matrix(X, y, w, b, logistic=True, lambda_=0.1)
print("Cost:", cost)
```

---

## 📦 Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.7 |
| NumPy | ≥ 1.21 |
| Matplotlib | ≥ 3.4 |

---

## 👤 Author

**Nikunj Thakur**
[![GitHub](https://img.shields.io/badge/GitHub-Nikunj--Thakur-181717?logo=github)](https://github.com/Nikunj-Thakur)
