import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline  # Note: imblearn's Pipeline to support SMOTE
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------------
# Data Loading & Cleaning
# -------------------------
df = pd.read_csv("data/Telco_Customer_Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df = df.drop('customerID', axis=1)

# -------------------------
# Define Features and Target
# -------------------------
target = 'Churn'
X = df.drop(target, axis=1)
y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify numeric and categorical features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# -------------------------
# Preprocessing Pipeline
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# -------------------------
# Build Model Pipeline
# -------------------------
# We include SMOTE to balance the classes and use XGBoost as our classifier.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# -------------------------
# Hyperparameter Tuning with GridSearchCV
# -------------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10],
    'classifier__learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

# Use the best model from grid search
best_model = grid_search.best_estimator_
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# -------------------------
# Adjust the Decision Threshold
# -------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
# Find threshold that maximizes F1 score
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(f1_scores)]
print("Optimal Threshold:", best_thresh)

# Apply the optimal threshold
y_pred_adjusted = (y_pred_prob >= best_thresh).astype(int)

# -------------------------
# Model Evaluation Metrics
# -------------------------
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision_val = precision_score(y_test, y_pred_adjusted)
recall_val = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("Adjusted Accuracy: {:.2f}".format(accuracy))
print("Adjusted Precision: {:.2f}".format(precision_val))
print("Adjusted Recall: {:.2f}".format(recall_val))
print("Adjusted F1 Score: {:.2f}".format(f1))
print("ROC-AUC: {:.2f}".format(roc_auc))

# -------------------------
# Visualization: Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred_adjusted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - XGBoost with SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("xgb_confusion_matrix.png", bbox_inches='tight')
plt.show()

# -------------------------
# Visualization: ROC Curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost with SMOTE")
plt.legend(loc="lower right")
plt.savefig("xgb_roc_curve.png", bbox_inches='tight')
plt.show()

