# Heart Disease Prediction using Logistic Regression

This repository implements a logistic regression model specifically for predicting heart disease risk using the `predictHeartDisease.csv` dataset.

**Introduction**

World Health Organization has estimated 12 million deaths occur worldwide every year due to heart diseases. Half the deaths in the United States and other developed countries are due to cardiovascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high-risk patients and in turn reduce the complications.

This research intends to pinpoint the most relevant risk factors for heart disease as well as predict the overall risk using logistic regression.

The project demonstrates:

- a from-scratch logistic regression implementation,
- stable logistic cost computation,
- gradient descent training,
- feature importance and risk prediction for heart disease,
- evaluation with precision, recall, F1-score, ROC-AUC, and threshold tuning.

NOTE: This project is intended for learning and experimentation — for production or larger datasets prefer `scikit-learn`'s `LogisticRegression`.

---

## Quick overview

- `heart_disease_prediction_model.py` — end-to-end example using `predictHeartDisease.csv` (normalisation, training, plotting, single-record prediction, and threshold-based evaluation).
- `logistic_utility_function.py` — core math: `sigmoid`, `log_1pexp`, cost, gradient and `gradient_descent`.
- `images/` — visual assets used in the README.
- `basic_plot/` — small plotting experiments.

---

## Visuals included (from images/)

- `Sigmoid_Function.png` — sigmoid / logistic activation curve
- `Decision_Boundary.png` — example 2D decision boundary (toy 2-feature examples only)
- `Logistic_Cost_Function.png` — logistic loss (convex) illustration
- `MSE_NonConvex_Function.png` — MSE composed with sigmoid (non-convex example)
- `Binary_Plots.png` — toy binary scatter plot
- `Feature_Importance.png` — bar chart of absolute learned weights
- `Feature_Histogram.png` — per-feature histograms split by class
- `Model_Convergence.png` — cost vs iterations convergence plot

Inline previews:

![Sigmoid Function](images/Sigmoid_Function.png)

![Decision Boundary (2D toy)](images/Decision_Boundary.png)

![Logistic Loss (convex)](images/Logistic_Cost_Function.png)

![MSE with Sigmoid (non-convex)](images/MSE_NonConvex_Function.png)

![Binary labelled data (toy)](images/Binary_Plots.png)

![Feature importance (learned weights)](images/Feature_Importance.png)

![Feature histograms by class](images/Feature_Histogram.png)

![Model convergence (cost vs iterations)](images/Model_Convergence.png)

---

## Project layout

```
Logistic Regression/
├── heart_disease_prediction_model.py
├── logistic_utility_function.py
├── basic_plot/
├── images/
└── predictHeartDisease.csv
```

---

## Implementation notes

- Normalisation: training features are z-score normalised (store `X_mean` and `X_std` and use them to scale any test/sample inputs before prediction).
- Numerical stability: `log_1pexp` is used when computing the log-loss to avoid overflow for large inputs.
- Gradient descent: a simple batch gradient descent is implemented; the gradient routine returns `(dj_db, dj_dw)` and the trainer applies updates accordingly.

Key functions in `logistic_utility_function.py`:

- `sigmoid(z)` — logistic activation
- `log_1pexp(x, maximum=20)` — numerically stable log(1+exp(x))
- `compute_cost_matrix(X, y, w, b, logistic=True, lambda_=0, safe=True)` — cost with optional L2 regularisation
- `compute_gradient_logistic(X, y, w, b)` — computes gradients returned as `(dj_db, dj_dw)`

---

## Running the example

From the `Logistic Regression` folder run:

```bash
python heart_disease_prediction_model.py
```

What the script does:

- Reads `predictHeartDisease.csv` and drops rows with missing values.
- Builds `X` by dropping the `TenYearCHD` column and `y` from `TenYearCHD`.
- Normalises `X`, trains via `gradient_descent`, and displays plots (feature importance, histograms and convergence).

Notes:

- If you regenerate plots programmatically, save them to `images/` with the exact filenames used here so the README previews remain valid.
- Any prediction using learned parameters must apply the same normalization used during training.

---

## Evaluation & class imbalance guidance

The example dataset contains approximately 15% positive examples (`TenYearCHD == 1`). With this imbalance:

- Overall accuracy can be misleading; prefer `precision`, `recall`, `F1-score`, and `ROC-AUC`.
- The current example evaluates several decision thresholds and reports how each changes the trade-off between false positives and false negatives.
- In the latest run, a threshold of `0.3` gave the best F1-score on the test set for this model.

Recommended quick approaches:

1. Use class weights (in scikit-learn or by applying per-sample weights in the gradient computation).
2. Oversample the minority class (SMOTE) or undersample the majority class.
3. Tune the decision threshold using ROC/PR curves to optimize the metric you care about.

Quick evaluation snippet (requires `scikit-learn`):

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X = df.drop(columns=['TenYearCHD']).values
y = df['TenYearCHD'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# normalize X_train and X_test using X_train mean/std, train the model on X_train_norm
# compute probabilities on X_test_norm: y_prob = sigmoid(X_test_norm.dot(w) + b)
# choose a threshold to get y_pred and then:
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_prob))
```

---

## Suggested next steps

1. Add a `train_eval.py` script to perform a proper `train_test_split`, evaluate metrics and save plots.
2. Implement `class_weight` support in the trainer to handle imbalance without resampling.
3. Tune the model further with regularisation and a wider threshold search.
4. Add automated tests that ensure the example script runs and key outputs (cost decreases, shapes match).

---

## Minimal dependencies

```bash
pip install numpy matplotlib pandas
# optional for evaluation: pip install scikit-learn
```

---

## Author

Nikunj Thakur

