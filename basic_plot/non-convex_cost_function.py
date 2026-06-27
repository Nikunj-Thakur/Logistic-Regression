import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Example training data
# ---------------------------------------------------

x = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0,  0, 0, 1, 1, 1, 0, 0, 1, 1])

m = len(x)

b = 10

# ---------------------------------------------------
# Cost function
# ---------------------------------------------------


def compute_cost(w, b):
    g = w * x + b
    predictions = 1/(1+np.exp(-g))
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


w_array = np.linspace(-10, 10, 500)
cost_array = []

for w in w_array:
    cost_array.append(compute_cost(w, b))

print(cost_array)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(w_array, cost_array, color='b')
ax1.set_xlabel("w")
ax1.set_ylabel("MSE Cost")
ax1.set_title("MSE Cost vs w, with b fixed to 10")
ax1.grid(True)

fig.text(
    0.5,                # x position (center)
    0.001,               # y position (bottom area, below plots)
    ("1. Composing MSE with the sigmoid produces a wiggly loss function full of local minima, so gradient descent can get stuck and never find the globally optimal weights. \n"
     "2. MSE penalizes a 51% wrong prediction almost the same as a 99% wrong prediction, but a confidently wrong prediction should be punished far more harshly — log loss grows to infinity as confidence in the wrong answer grows.\n"
     "3. MSE is derived from the assumption that errors are Gaussian (continuous), but binary labels follow a Bernoulli distribution; log loss is the statistically correct loss derived directly from Maximum Likelihood Estimation for that distribution.\n"
     ),

    ha='center',
    fontsize=10,
    wrap=True,
    color='darkblue',
    bbox=dict(
        facecolor='lightyellow',
        edgecolor='black',
        boxstyle='round,pad=0.5',
        linewidth=2
    )
)
plt.show()
