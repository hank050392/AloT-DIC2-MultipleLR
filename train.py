import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("50_Startups.csv")

print("===== Dataset =====")
print(df.head())

print("\n===== Dataset Information =====")
print(df.info())


# ==========================================
# 2. Features and Target
# ==========================================

X = df[
    [
        "R&D Spend",
        "Administration",
        "Marketing Spend",
        "State"
    ]
]

y = df["Profit"]


# ==========================================
# 3. Data Preprocessing
# ==========================================

numeric_features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

categorical_features = [
    "State"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "state",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 4. Build Multiple Linear Regression Model
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==========================================
# 5. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. Training
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 7. Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n===== Model Evaluation =====")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 9. Actual vs Predicted Results
# ==========================================

results = pd.DataFrame({
    "Actual Profit": y_test.values,
    "Predicted Profit": y_pred
})

results = results.sort_values(
    by="Actual Profit"
).reset_index(drop=True)

print("\n===== Prediction Results =====")
print(results)


# ==========================================
# 10. Create Images Folder
# ==========================================

os.makedirs("images", exist_ok=True)


# ==========================================
# 11. Actual vs Predicted Curve
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    results.index,
    results["Actual Profit"],
    marker="o",
    label="Actual Profit"
)

plt.plot(
    results.index,
    results["Predicted Profit"],
    marker="o",
    label="Predicted Profit"
)

plt.title("Actual Profit vs Predicted Profit")

plt.xlabel("Sample")
plt.ylabel("Profit")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ==========================================
# 12. R&D Spend vs Profit Regression Line
# ==========================================

X_rd = df[["R&D Spend"]].values
y_profit = df["Profit"].values

rd_model = LinearRegression()
rd_model.fit(X_rd, y_profit)

# 產生平滑的 X 軸資料
x_line = np.linspace(
    X_rd.min(),
    X_rd.max(),
    200
).reshape(-1, 1)

# 預測 Regression Line
y_line = rd_model.predict(x_line)

plt.figure(figsize=(10, 6))

# 真實資料
plt.scatter(
    X_rd,
    y_profit,
    label="Actual Data"
)

# Regression Line
plt.plot(
    x_line,
    y_line,
    label="Regression Line"
)

plt.title("R&D Spend vs Profit")

plt.xlabel("R&D Spend")
plt.ylabel("Profit")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/rd_spend_vs_profit.png",
    dpi=300
)

plt.show()

# ==========================================
# 13. Save Model
# ==========================================

joblib.dump(
    model,
    "startup_profit_model.pkl"
)

print("\nModel saved as startup_profit_model.pkl")