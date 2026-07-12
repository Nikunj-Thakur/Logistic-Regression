import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

df = pd.read_csv("predictHeartDisease.csv")
df = df.dropna()  # Drop rows with missing values before training.

# Separate features (X) from the target (y).
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Fit normalization 
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0  # avoid division by zero for constant features
X_norm = (X - X_mean) / X_std

print(
    f"Peak to Peak range by column in Raw        X:\n{np.ptp(X, axis=0)}")
print(
    f"Peak to Peak range by column in Normalized X:\n{np.ptp(X_norm, axis=0)}")

lr_model = LogisticRegression()
lr_model.fit(X_norm, y)
y_pred = lr_model.predict(X_norm)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X_norm, y))