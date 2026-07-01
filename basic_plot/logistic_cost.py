import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logistic_utility_function as luf

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)

def plt_logistic_cost(X,y):
    """ plots logistic cost """
    wx, by = np.meshgrid(np.linspace(-6,12,50),
                         np.linspace(0, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0],dtype=np.longdouble)

    for i in range(points.shape[0]):
        w,b = points[i]
        cost[i] = luf.compute_cost_matrix(X.reshape(-1,1), y, w, b, logistic=True, safe=True)
    cost = cost.reshape(wx.shape)

    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(wx, by, cost, alpha=0.6,cmap=cm.jet)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('Logistic Cost vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(wx, by, np.log(cost), alpha=0.6,cmap=cm.jet)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel('\nlog(Cost)', fontsize=16)
    ax.set_title('log(Logistic Cost) vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    #This curve is well suited to gradient descent! It does not have plateaus, local minima, or discontinuities. 

    fig.text(
    0.5,                # x position (center)
    0.1,               # y position (bottom area, below plots)
    ("This curve is well suited to gradient descent! It does not have plateaus, local minima, or discontinuities.\n"
    "Note, it is not a bowl as in the case of squared error.\n"
    "Both the cost and the log of the cost are plotted to illuminate the fact that the curve,\n "
    "when the cost is small, has a slope and continues to decline."),

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
    return cost


cst = plt_logistic_cost(x_train,y_train)