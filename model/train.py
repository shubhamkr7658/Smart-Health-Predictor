import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# ======================================================
# 1. LOAD DATA
# ======================================================
CSV_PATH = "Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv"
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# ======================================================
# 2. TARGET HANDLING
# ======================================================
if "health_risk" in df.columns:
    if df["health_risk"].dtype == object:
        mapping = {
            "very low": 10, "low": 25,
            "medium": 50, "high": 75,
            "very high": 90
        }
        df["health_risk"] = df["health_risk"].map(lambda x: mapping.get(str(x).lower(), 50))
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["health_risk"] = df[numeric_cols].sum(axis=1)
    df["health_risk"] = 100 * (df["health_risk"] - df["health_risk"].min()) / (df["health_risk"].max() - df["health_risk"].min())

target = "health_risk"
X = df.drop(columns=[target])
y = df[target].astype(float)

# ======================================================
# 3. FEATURE TYPES
# ======================================================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ======================================================
# 4. PREPROCESSING
# ======================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ======================================================
# 5. TRAIN-TEST SPLIT (FIXED - OUTSIDE LOOP)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 6. PIPELINE + GRID SEARCH (INDUSTRY STANDARD)
# ======================================================
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

param_grid = {
    "regressor__n_estimators": [200, 300],
    "regressor__max_depth": [10, 15],
    "regressor__min_samples_split": [2, 5],
    "regressor__min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

print("\nTraining model with cross-validation...\n")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ======================================================
# 7. EVALUATION
# ======================================================
pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("\nTraining Completed!")
print("Best Parameters:", grid.best_params_)
print("MAE:", mae)

# ======================================================
# 8. FEATURE IMPORTANCE (INTERVIEW GOLD 🔥)
# ======================================================
model = best_model.named_steps["regressor"]

feature_names = (
    numeric_features +
    list(best_model.named_steps["preprocessor"]
         .transformers_[1][1]
         .named_steps["onehot"]
         .get_feature_names_out(categorical_features))
)

importances = model.feature_importances_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\nTop Important Features:\n")
print(feat_imp.head(10))

# ======================================================
# 9. USER INPUT
# ======================================================
print("\n============================")
print(" ENTER USER DETAILS BELOW ")
print("============================\n")

user_input = {}

for col in numeric_features:
    user_input[col] = float(input(f"Enter {col}: "))

for col in categorical_features:
    user_input[col] = input(f"Enter {col}: ")

user_df = pd.DataFrame([user_input])

# ======================================================
# 10. PREDICTION
# ======================================================
prediction = best_model.predict(user_df)[0]
prediction = max(0, min(100, prediction))

print("\n==================================")
print(f"Predicted Health Risk: {prediction:.2f} / 100")
print("==================================\n")

import joblib
import os

# Save model safely inside model folder
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(best_model, model_path)

print("✅ Model saved successfully!")

# ======================================================
