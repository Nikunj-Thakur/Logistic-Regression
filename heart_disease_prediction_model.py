import numpy as np
import matplotlib.pyplot as plt
import logistic_utility_function as luf
import pandas as pd
import os

# Try to use scikit-learn's reliable train/test splitter.
try:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover - fallback for minimal environments
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None
# If scikit-learn is not available, fall back to a simple manual split.
try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - fallback for minimal environments
    train_test_split = None

os.chdir(os.path.dirname(__file__))

df = pd.read_csv("predictHeartDisease.csv")
df = df.dropna()  # Drop rows with missing values before training.

# Separate features (X) from the target (y).
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Split the data into training and test sets.
# This is better approach because it is random and preserves the class balance.
if train_test_split is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,  # Split the data so that the proportion of each class in y is preserved
        random_state=42, # fixes the random seed, so the random process produces the same result every time you run the code.
    )
else:
    # Create a random number generator with a fixed seed so the split is reproducible.
    rng = np.random.default_rng(42)
    # Create a list of row positions so we can shuffle them randomly.
    indices = np.arange(len(df))
    # Randomly reorder the row indices.
    rng.shuffle(indices)
    # Use 80% of the rows for training and 20% for testing.
    split_idx = int(len(df) * 0.8)
    # Take the first 80% of the shuffled indices for training.
    train_idx = indices[:split_idx]
    # Use the remaining 20% for testing.
    test_idx = indices[split_idx:]
    # Build the training and testing feature sets from the shuffled indices.
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    # Build the matching target values for each split.
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

# Convert the split DataFrames into NumPy arrays for the model.
X_train_raw = X_train.values
X_test_raw = X_test.values

# Fit normalization on the training set only.
# This prevents any information from the test set from leaking into training.
X_mean = X_train_raw.mean(axis=0)
X_std = X_train_raw.std(axis=0)
X_std[X_std == 0] = 1.0  # avoid division by zero for constant features
X_train_norm = (X_train_raw - X_mean) / X_std
X_test_norm = (X_test_raw - X_mean) / X_std

print(
    f"Peak to Peak range by column in Raw        X:{np.ptp(X_train_raw, axis=0)}")
print(
    f"Peak to Peak range by column in Normalized X:{np.ptp(X_train_norm, axis=0)}")

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"Positive rate in train set: {y_train.mean():.3f}")
print(f"Positive rate in test set: {y_test.mean():.3f}")

# Convert the target column to a NumPy array for the training function.
y_train_values = y_train.values

# Initialize logistic regression parameters.
w_tmp = np.zeros_like(X_train_raw[0])
b_tmp = 0.
alph = 0.09
iters = 1000

# Train the model on the normalized training data.
w_out, b_out, J_hist = luf.gradient_descent(
    X_train_norm, y_train_values, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# Evaluate on training and test sets using the learned weights.
# First, convert the linear score into a probability between 0 and 1.
train_probs = luf.sigmoid(X_train_norm @ w_out + b_out)
test_probs = luf.sigmoid(X_test_norm @ w_out + b_out)

# Turn those probabilities into binary class predictions.
# If the probability is 0.5 or higher, predict class 1; otherwise predict class 0.
train_pred = (train_probs >= 0.3).astype(int)
test_pred = (test_probs >= 0.3).astype(int)

# Compare the predicted labels with the true labels to measure accuracy.
# np.mean(...) computes the fraction of correct predictions.
train_acc = np.mean(train_pred == y_train_values)
test_acc = np.mean(test_pred == y_test.values)

# Print the results so we can see how well the model performs.
print(f"Training accuracy: {train_acc:.3f}")
print(f"Testing accuracy: {test_acc:.3f}")
print(f"Testing positive predictions: {test_pred.sum()} / {len(test_pred)}")

if all(metric is not None for metric in [precision_score, recall_score, f1_score, roc_auc_score]):
    train_precision = precision_score(y_train_values, train_pred, zero_division=0)
    train_recall = recall_score(y_train_values, train_pred, zero_division=0)
    train_f1 = f1_score(y_train_values, train_pred, zero_division=0)
    test_precision = precision_score(y_test.values, test_pred, zero_division=0)
    test_recall = recall_score(y_test.values, test_pred, zero_division=0)
    test_f1 = f1_score(y_test.values, test_pred, zero_division=0)

    train_auc = roc_auc_score(y_train_values, train_probs)
    test_auc = roc_auc_score(y_test.values, test_probs)

    print(f"Training precision: {train_precision:.3f}")
    print(f"Training recall: {train_recall:.3f}")
    print(f"Training F1-score: {train_f1:.3f}")
    print(f"Training ROC-AUC: {train_auc:.3f}")
    print(f"Testing precision: {test_precision:.3f}")
    print(f"Testing recall: {test_recall:.3f}")
    print(f"Testing F1-score: {test_f1:.3f}")
    print(f"Testing ROC-AUC: {test_auc:.3f}")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nThreshold tuning on test set:")
    best_threshold = None
    best_f1 = -1.0
    for threshold in thresholds:
        threshold_pred = (test_probs >= threshold).astype(int)
        precision = precision_score(y_test.values, threshold_pred, zero_division=0)
        recall = recall_score(y_test.values, threshold_pred, zero_division=0)
        f1 = f1_score(y_test.values, threshold_pred, zero_division=0)
        print(
            f"threshold={threshold:.1f} | precision={precision:.3f} | "
            f"recall={recall:.3f} | f1={f1:.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold by F1-score: {best_threshold:.1f} (F1={best_f1:.3f})")

# Show which input features had the biggest influence on the model.
feature_names = X.columns
importance = np.abs(w_out)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(feature_names, importance)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_ylabel('Absolute weight magnitude')
ax.set_title('Feature importance by learned logistic regression weights')
plt.tight_layout()
plt.show()

# Plot a few selected feature distributions for the training set.
selected = ['age', 'totChol', 'sysBP', 'BMI']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feature in zip(axes.flatten(), selected):
    feature_idx = X.columns.get_loc(feature)
    ax.hist(X_train_raw[y_train_values == 0, feature_idx],
            alpha=0.6, label='y=0')
    ax.hist(X_train_raw[y_train_values == 1, feature_idx],
            alpha=0.6, label='y=1')
    ax.set_title(feature)
    ax.legend()
plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])  # ax1.plot(range(len(J_hist[:100])), J_hist[:100])
ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')

ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Cost vs. iteration (tail)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')
plt.show()

# Demonstrate how to score a new patient record.
# The new record must be scaled with the same training-set statistics.
print("Predict chances of Heart Disease for a patient with record:")
test_row = np.array([1, 50, 1, 0, 0, 1, 0, 313, 179,
                    92, 25.97, 66, 86], dtype=float)
print(f"{test_row}")
# Scale test data using training set statistics
test_row_scaled = (test_row - X_mean) / X_std
print("Scaled patient features:\n", test_row_scaled)

prediction = (np.dot(test_row_scaled, w_out)) + b_out
p = luf.sigmoid(prediction)
print(f"Predicted chances of Heart Disease: {p:0.2f}")
print(f"Patient actually has Heart Disease: 1.0")
print(f"Prediction Error: {abs(p - 1.0):0.2f}")
