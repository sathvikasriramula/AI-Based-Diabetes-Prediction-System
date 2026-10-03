import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib 
 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
 
from sklearn.metrics import ( 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve 
) 
 
DATASET_PATH = "diabetes_health_dataset_clean.csv" 
 
df = pd.read_csv(DATASET_PATH) 
 
print("\nDataset loaded successfully!") 
print("Shape:", df.shape) 
 
print("\nFirst 5 rows:") 
print(df.head()) 
 
print("\nColumn names:") 
print(df.columns.tolist()) 
 
print("\nDataset Information:") 
print(df.info()) 
 
print("\nMissing Values:") 
print(df.isnull().sum()) 
 
print("\nDuplicate Rows:", df.duplicated().sum()) 
 
df = df.drop_duplicates() 
 
possible_targets = [ 
    "Diabetes_binary", 
    "diabetes_binary", 
    "Diabetes", 
    "diabetes", 
    "Outcome", 
    "outcome" 
] 
 
target_column = None 
 
for column in possible_targets: 
    if column in df.columns: 
        target_column = column 
        break 
 
if target_column is None: 
    raise ValueError( 
        "Target column not found. Please set target_column manually." 
    ) 
 
print("\nTarget Column:", target_column) 
 
X = df.drop(columns=[target_column]) 
y = df[target_column] 
 
y = y.astype(int) 
 
print("\nTarget Distribution:") 
print(y.value_counts()) 
 
categorical_columns = X.select_dtypes( 
    include=["object", "category"] 
).columns 
 
if len(categorical_columns) > 0: 
    X = pd.get_dummies( 
        X, 
        columns=categorical_columns, 
        drop_first=True 
    ) 
 
imputer = SimpleImputer(strategy="median") 
 
X_imputed = imputer.fit_transform(X) 
 
X = pd.DataFrame( 
    X_imputed, 
    columns=X.columns 
) 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y 
) 
 
print("\nTraining data:", X_train.shape) 
print("Testing data:", X_test.shape) 
 
scaler = StandardScaler() 
 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
 
model = RandomForestClassifier( 
    n_estimators=200, 
    random_state=42, 
    class_weight="balanced" 
) 
 
model.fit(X_train_scaled, y_train) 
 
print("\nRandom Forest model trained successfully!") 
 
y_pred = model.predict(X_test_scaled) 
 
y_probability = model.predict_proba(X_test_scaled)[:, 1] 
 
accuracy = accuracy_score(y_test, y_pred) 
 
precision = precision_score( 
    y_test, 
    y_pred, 
    zero_division=0 
) 
 
recall = recall_score( 
    y_test, 
    y_pred, 
    zero_division=0 
) 
 
f1 = f1_score( 
    y_test, 
    y_pred, 
    zero_division=0 
) 
 
roc_auc = roc_auc_score( 
    y_test, 
    y_probability 
) 
 
print("\n========================================") 
print("MODEL EVALUATION") 
print("========================================") 
 
print(f"Accuracy  : {accuracy:.4f}") 
print(f"Precision : {precision:.4f}") 
print(f"Recall    : {recall:.4f}") 
print(f"F1-Score  : {f1:.4f}") 
print(f"ROC-AUC   : {roc_auc:.4f}") 
 
print("\nClassification Report:") 
print(classification_report( 
    y_test, 
    y_pred, 
    zero_division=0 
)) 
 
cm = confusion_matrix(y_test, y_pred) 
 
plt.figure(figsize=(6, 5)) 
 
sns.heatmap( 
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=["Non-Diabetic", "Diabetic"], 
    yticklabels=["Non-Diabetic", "Diabetic"] 
) 
 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.title("Confusion Matrix") 
 
plt.tight_layout() 
plt.savefig("confusion_matrix.png") 
plt.show() 
 
fpr, tpr, thresholds = roc_curve( 
    y_test, 
    y_probability 
) 
 
plt.figure(figsize=(7, 5)) 
 
plt.plot( 
    fpr, 
    tpr, 
    label=f"Random Forest (AUC = {roc_auc:.3f})" 
) 
 
plt.plot( 
    [0, 1], 
    [0, 1], 
    linestyle="--" 
) 
 
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.title("ROC-AUC Curve") 
plt.legend() 
 
plt.tight_layout() 
plt.savefig("roc_auc_curve.png") 
plt.show() 
 
feature_importance = pd.DataFrame({ 
    "Feature": X.columns, 
    "Importance": model.feature_importances_ 
}) 
 
feature_importance = feature_importance.sort_values( 
    by="Importance", 
    ascending=False 
) 
 
print("\nTop Important Features:") 
print(feature_importance.head(15)) 
 
plt.figure(figsize=(10, 7)) 
 
sns.barplot( 
    data=feature_importance.head(15), 
    x="Importance", 
    y="Feature" 
) 
 
plt.title("Top 15 Feature Importances") 
plt.tight_layout() 
 
plt.savefig("feature_importance.png") 
plt.show() 
 
joblib.dump(model, "diabetes_model.pkl") 
joblib.dump(scaler, "scaler.pkl") 
joblib.dump(imputer, "imputer.pkl") 
joblib.dump(X.columns.tolist(), "feature_columns.pkl") 
 
print("\n========================================") 
print("Files saved successfully!") 
print("diabetes_model.pkl") 
print("scaler.pkl") 
print("imputer.pkl") 
print("feature_columns.pkl") 
print("========================================")