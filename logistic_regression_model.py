import numpy as np
import matplotlib.pyplot as plt
import logistic_utility_function as luf

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = luf.gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
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
ax.set_title('two variable plot')
plt.tight_layout()

x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0])
plt.show()
plt.show()