import numpy as np
import matplotlib.pyplot as plt
import logistic_utility_function as luf
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

df = pd.read_csv("predictHeartDisease.csv")
df = df.dropna()  # dropping small percentage of rows have missing values

X_train = df.drop(columns=['TenYearCHD']).values  # Features: all columns except target
X_mean = X_train.mean(axis=0)  # store them for normalizing test data later on
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1.0  # avoid division by zero for constant features
X_norm = (X_train - X_mean) / X_std

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm, axis=0)}")

y_train = df['TenYearCHD'].values  # Target variable

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.01
iters = 5000

w_out, b_out, J_hist = luf.gradient_descent(X_norm, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

feature_names = df.drop(columns=['TenYearCHD']).columns
importance = np.abs(w_out)
  
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(feature_names, importance)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_ylabel('Absolute weight magnitude')
ax.set_title('Feature importance by learned logistic regression weights')
plt.tight_layout()
plt.show()

selected = ['age', 'totChol', 'sysBP', 'BMI']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feature in zip(axes.flatten(), selected):
    ax.hist(X_train[y_train == 0, df.columns.get_loc(feature)], alpha=0.6, label='y=0')
    ax.hist(X_train[y_train == 1, df.columns.get_loc(feature)], alpha=0.6, label='y=1')
    ax.set_title(feature)
    ax.legend()
plt.tight_layout()
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

print("Predict chances of Heart Disease for a patient with record:")
test_row = np.array([1,56,0,1,0,1,0,287,149,98,21.68,90,75],dtype=float)
print(f"{test_row}")
# Scale test data using training set statistics
test_row_scaled = (test_row - X_mean) / X_std
print("Scaled patient features:\n", test_row_scaled)

prediction = (np.dot(test_row_scaled, w_out)) + b_out
p=luf.sigmoid(prediction)
print(f"Predicted chances of Heart Disease: {p:0.2f}")
print(f"Patient actually has Heart Disease: 1.0")
print(f"Prediction Error: {abs(p - 1.0):0.2f}")
