import numpy as np
import matplotlib.pyplot as plt
import logistic_utility_function as luf

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, J_hist = luf.gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")


pos = y_train == 1
neg = y_train == 0
pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
neg = neg.reshape(-1,)

fig,ax = plt.subplots(figsize=(8,3))
ax.scatter(X_train[pos, 0], X_train[pos, 1], marker='x')
ax.scatter(X_train[neg, 0], X_train[neg, 1], marker='o')

ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
ax.set_title('Two variable plot')
plt.tight_layout()

x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0])
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100]) # ax1.plot(range(len(J_hist[:100])), J_hist[:100])
ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')

ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Cost vs. iteration (tail)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')

plt.show()