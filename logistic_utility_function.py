import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def log_1pexp(x, maximum=20):
    ''' approximate log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    Args:
    x   : (ndarray Shape (n,1) or (n,)  input
    out : (ndarray Shape matches x      output ~= np.log(1+exp(x))
    '''

    output = np.zeros_like(x, dtype=float)
    i = x <= maximum  # e.g. [True,True,True,False]
    ni = np.logical_not(i)  # e.g. [False,False,False,True]

    output[i] = np.log(1 + np.exp(x[i]))
    output[ni] = x[ni]
    return output


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1/(1+np.exp(-z))

    return g


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    Computes the cost using  using matrices
    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameter(s) of the model
      b : (scalar )                       Values of parameter of the model
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns:
      total_cost: (scalar)                cost
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)             # ensure 2D
    w = w.reshape(-1, 1)             # ensure 2D
    if logistic:
        if safe:  # safe from overflow
            z = X @ w + b  # (m,n)(n,1)=(m,1)
            cost = -(y * z) + log_1pexp(z)
            # (scalar)
            cost = np.sum(cost)/m
        else:
            # (m,n)(n,1) = (m,1)
            f = sigmoid(X @ w + b)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) -
                          np.dot((1-y).T, np.log(1-f)))   # (1,m)(m,1) = (1,1)
            # scalar
            cost = cost[0, 0]
    else:
        # (m,n)(n,1) = (m,1)
        f = X @ w + b
        # scalar
        cost = (1/(2*m)) * np.sum((f - y)**2)

    # scalar
    reg_cost = (lambda_/(2*m)) * np.sum(w**2)

    total_cost = cost + reg_cost                                                # scalar

    return total_cost


def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression 

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m  # (n,)
    dj_db = dj_db/m  # scalar

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save the logistic cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(compute_cost_matrix(X, y, w, b, logistic=True))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing
