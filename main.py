# ==============================
# Credit Card Fraud Detection
# ==============================

# 1️⃣ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

# 2️⃣ Load Dataset
data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print("\nClass Distribution:\n", data["Class"].value_counts())

# 3️⃣ Feature & Target Split
X = data.drop("Class", axis=1)
Y = data["Class"]

# 4️⃣ Scale 'Amount' feature (Important)
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# 5️⃣ Train-Test Split (Stratified)
xTrain, xTest, yTrain, yTest = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("\nTraining shape:", xTrain.shape)
print("Testing shape:", xTest.shape)

# 6️⃣ Train Random Forest (with class balancing)
rfc = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rfc.fit(xTrain, yTrain)

# 7️⃣ Predictions
yPred = rfc.predict(xTest)
yProb = rfc.predict_proba(xTest)[:, 1]

# ==============================
# 8️⃣ Evaluation
# ==============================

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy :", accuracy_score(yTest, yPred))
print("Precision:", precision_score(yTest, yPred))
print("Recall   :", recall_score(yTest, yPred))
print("F1 Score :", f1_score(yTest, yPred))
print("ROC-AUC  :", roc_auc_score(yTest, yProb))

print("\nClassification Report:\n")
print(classification_report(yTest, yPred))

# ==============================
# 9️⃣ Confusion Matrix
# ==============================

conf_matrix = confusion_matrix(yTest, yPred)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Valid", "Fraud"],
            yticklabels=["Valid", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 🔟 ROC Curve
# ==============================

fpr, tpr, thresholds = roc_curve(yTest, yProb)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ==============================
# 1️⃣1️⃣ Feature Importance
# ==============================

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()