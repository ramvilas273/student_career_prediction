import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Create models folder if not exists
# ===============================
os.makedirs("models", exist_ok=True)

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("C:\\Users\\ramvi\\Downloads\\Student-Career-Prediction\\cs_students.csv")

print("Original Dataset Shape:", df.shape)

# ===============================
# Drop Unnecessary Columns
# ===============================
df = df.drop(["Student ID", "Name", "Interested Domain"], axis=1)

# ===============================
# Remove Rare Classes (<2 samples)
# ===============================
class_counts = df["Future Career"].value_counts()
valid_classes = class_counts[class_counts > 1].index
df = df[df["Future Career"].isin(valid_classes)]

print("After Removing Rare Classes:", df.shape)

# ===============================
# Encode Features
# ===============================
feature_encoders = {}

categorical_cols = ["Gender", "Major", "Projects", "Python", "SQL", "Java"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    feature_encoders[col] = le

# ===============================
# Encode Target
# ===============================
target_encoder = LabelEncoder()
df["Future Career"] = target_encoder.fit_transform(df["Future Career"])

# ===============================
# Define Features and Target
# ===============================
X = df.drop("Future Career", axis=1)
y = df["Future Career"]

model_columns = X.columns.tolist()

print("Final Feature Columns:", model_columns)

# ===============================
# Train-Test Split (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Scaling
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Hyperparameter Tuning
# ===============================
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# ===============================
# Evaluation
# ===============================
y_pred = best_model.predict(X_test_scaled)

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# Save Model and Preprocessing Objects
# ===============================
joblib.dump(best_model, "models/career_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_encoders, "models/feature_encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")
joblib.dump(model_columns, "models/model_columns.pkl")

print("\n✅ Model and preprocessing objects saved successfully!")