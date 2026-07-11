import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1  # Example: pos = [False, False, False, True, True, True]
    neg = y == 0  # Example: neg = [True, True, True, False, False, False]
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label) 
     # X[pos, 0] : Take rows where pos=True, Take column 0 (x1)
     # X[pos, 1] : Take rows where pos=True, Take column 1 (x1)


    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors='b', lw=3)
     # X[neg, 0] : Take rows where neg=True, Take column 0 (x1)
     # X[neg, 1] : Take rows where neg=True, Take column 1 (x1)

    ax.legend(loc=loc)



x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0

fig,ax = plt.subplots(1,2,figsize=(8,3))
#plot 1, single variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors='b',lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()


fig.text(
    0.5,                # x position (center)
    0.10,               # y position (bottom area, below plots)
    ("In the two-variable plot, the y axis is not available. Positive results are shown as red 'X's, "
    "while negative results use the blue 'O' symbol. "
    "Recall in the case of linear regression with multiple variables, "
    "y would not have been limited to two values and a similar plot would have been three-dimensional."),

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