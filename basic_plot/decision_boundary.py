import numpy as np
import matplotlib.pyplot as plt


X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, )

pos = y == 1
neg = y == 0

print(pos)
print(neg)

x0=np.arange(0,6)
x1=3-x0

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.scatter(X[pos, 0], X[pos, 1], marker='x', color='r',label="y=1")
ax.scatter(X[neg, 0], X[neg, 1], marker='o', color="b",label="y=0")
ax.plot(x0,x1)

ax.axis([0, 4, 0, 3.5])
ax.set_title("Decision Boundary")
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.legend()

fig.text(
    0.5,                # x position (center)
    0.01,               # y position (bottom area, below plots)
    ("In the plot above, the blue line represents the line x0+x1=3\n"
     "Any point under the line is classified as y=0. Any point on or above the line is classified as y=1\n"
    "This line is known as the 'Decision boundary'"),

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
