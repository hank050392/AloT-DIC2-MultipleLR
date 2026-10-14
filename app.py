import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# Page Setting
# =========================================================

st.set_page_config(
    page_title="Regression Analysis System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Regression Analysis System")

st.write(
    """
    This system demonstrates regression analysis using
    Multiple Linear Regression.

    You can choose between two regression problems
    or enter your own CSV dataset URL.
    """
)


# =========================================================
# Problem Selection
# =========================================================

st.sidebar.header("Settings")

problem = st.sidebar.selectbox(
    "Select Problem",
    [
        "50 Startups",
        "Boston Housing"
    ]
)


# =========================================================
# Default Dataset URLs
# =========================================================

STARTUP_URL = (
    "https://raw.githubusercontent.com/"
    "Avik-Jain/100-Days-Of-ML-Code/master/"
    "datasets/50_Startups.csv"
)

BOSTON_URL = (
    "https://raw.githubusercontent.com/"
    "hotcode002/Machine-Learning-with-Python/"
    "master/week_2_math/data/boston_housing.csv"
)


if problem == "50 Startups":

    default_url = STARTUP_URL

else:

    default_url = BOSTON_URL


# =========================================================
# Dataset Source
# =========================================================

st.sidebar.subheader("Dataset Source")

dataset_mode = st.sidebar.radio(
    "Choose Dataset",
    [
        "Default Dataset",
        "Custom Dataset URL"
    ]
)


if dataset_mode == "Default Dataset":

    dataset_url = default_url

    st.sidebar.info(
        "Using the default dataset for this problem."
    )

else:

    dataset_url = st.sidebar.text_input(
        "Enter CSV URL",
        value=default_url
    )


# =========================================================
# Load Dataset
# =========================================================

st.header("1. Dataset")

try:

    df = pd.read_csv(dataset_url)

except Exception as e:

    st.error("Failed to load dataset.")

    st.write(e)

    st.stop()


st.success(
    f"Dataset loaded successfully: {len(df)} rows"
)


# =========================================================
# Dataset Preview
# =========================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


# =========================================================
# Select Target
# =========================================================

st.header("2. Feature Selection")


# ---------------------------------------------------------
# Default target according to selected problem
# ---------------------------------------------------------

if dataset_mode == "Default Dataset":

    if problem == "50 Startups":

        default_target = "Profit"

    else:

        # Boston dataset usually uses medv
        if "medv" in df.columns:

            default_target = "medv"

        elif "MEDV" in df.columns:

            default_target = "MEDV"

        elif "PRICE" in df.columns:

            default_target = "PRICE"

        else:

            default_target = df.columns[-1]

else:

    default_target = df.columns[-1]


default_target_index = list(df.columns).index(
    default_target
)


target_column = st.selectbox(
    "Target Column",
    df.columns,
    index=default_target_index
)


available_features = [
    col
    for col in df.columns
    if col != target_column
]


selected_features = st.multiselect(
    "Feature Columns",
    available_features,
    default=available_features
)


if len(selected_features) == 0:

    st.warning(
        "Please select at least one feature."
    )

    st.stop()


# =========================================================
# Prepare Data
# =========================================================

X = df[selected_features].copy()

y = df[target_column].copy()


# Target must be numeric
y = pd.to_numeric(
    y,
    errors="coerce"
)


valid_rows = y.notna()

X = X.loc[valid_rows]

y = y.loc[valid_rows]


# =========================================================
# Detect Numeric and Categorical Columns
# =========================================================

numeric_features = (
    X.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


categorical_features = [
    col
    for col in selected_features
    if col not in numeric_features
]


st.write(
    "**Numeric Features:**",
    numeric_features
)

st.write(
    "**Categorical Features:**",
    categorical_features
)


# =========================================================
# Preprocessing
# =========================================================

transformers = []


if categorical_features:

    transformers.append(
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="passthrough"
)


# =========================================================
# Model
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# =========================================================
# Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# Training
# =========================================================

try:

    model.fit(
        X_train,
        y_train
    )

except Exception as e:

    st.error(
        "Model training failed."
    )

    st.write(e)

    st.stop()


# =========================================================
# Prediction
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# Model Evaluation
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


st.header("3. Model Evaluation")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "MAE",
        f"{mae:,.2f}"
    )


with col2:

    st.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )


with col3:

    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )


# =========================================================
# Actual vs Predicted
# =========================================================

st.header("4. Visualization")


results = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": y_pred
    }
)


results = results.sort_values(
    by="Actual"
).reset_index(
    drop=True
)


fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)


ax1.plot(
    results.index,
    results["Actual"],
    marker="o",
    label="Actual"
)


ax1.plot(
    results.index,
    results["Predicted"],
    marker="o",
    label="Predicted"
)


ax1.set_title(
    "Actual vs Predicted"
)

ax1.set_xlabel(
    "Sample"
)

ax1.set_ylabel(
    target_column
)

ax1.legend()

ax1.grid(True)


st.pyplot(fig1)


# =========================================================
# Regression Line
# =========================================================

if len(numeric_features) > 0:

    st.subheader(
        "Regression Relationship"
    )


    # Find numeric feature with strongest correlation
    correlation_data = df[
        numeric_features
        + [target_column]
    ].copy()


    correlation_data[
        target_column
    ] = pd.to_numeric(
        correlation_data[target_column],
        errors="coerce"
    )


    correlation_matrix = (
        correlation_data
        .corr(numeric_only=True)
    )


    correlation_series = (
        correlation_matrix[
            target_column
        ]
        .drop(
            target_column,
            errors="ignore"
        )
        .abs()
    )


    if len(correlation_series) > 0:

        best_feature = (
            correlation_series
            .idxmax()
        )


        regression_data = df[
            [
                best_feature,
                target_column
            ]
        ].copy()


        regression_data[
            best_feature
        ] = pd.to_numeric(
            regression_data[
                best_feature
            ],
            errors="coerce"
        )


        regression_data[
            target_column
        ] = pd.to_numeric(
            regression_data[
                target_column
            ],
            errors="coerce"
        )


        regression_data = (
            regression_data
            .dropna()
        )


        x_simple = (
            regression_data[
                [best_feature]
            ]
            .values
        )


        y_simple = (
            regression_data[
                target_column
            ]
            .values
        )


        simple_model = (
            LinearRegression()
        )


        simple_model.fit(
            x_simple,
            y_simple
        )


        x_line = np.linspace(
            x_simple.min(),
            x_simple.max(),
            200
        ).reshape(
            -1,
            1
        )


        y_line = simple_model.predict(
            x_line
        )


        fig2, ax2 = plt.subplots(
            figsize=(9, 5)
        )


        ax2.scatter(
            x_simple,
            y_simple,
            label="Actual Data"
        )


        ax2.plot(
            x_line,
            y_line,
            label="Regression Line"
        )


        ax2.set_title(
            f"{best_feature} vs {target_column}"
        )


        ax2.set_xlabel(
            best_feature
        )


        ax2.set_ylabel(
            target_column
        )


        ax2.legend()

        ax2.grid(True)


        st.pyplot(fig2)


        st.info(
            f"Feature with strongest correlation: "
            f"{best_feature}"
        )


# =========================================================
# User Prediction
# =========================================================

st.header("5. Make a Prediction")

st.write(
    "Enter feature values below:"
)


input_values = {}


columns = st.columns(2)


for index, feature in enumerate(
    selected_features
):

    column = columns[
        index % 2
    ]


    with column:

        if feature in numeric_features:

            default_value = float(
                pd.to_numeric(
                    df[feature],
                    errors="coerce"
                ).mean()
            )


            input_values[
                feature
            ] = st.number_input(
                feature,
                value=default_value
            )

        else:

            options = (
                df[feature]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            input_values[
                feature
            ] = st.selectbox(
                feature,
                options
            )


# =========================================================
# Predict Button
# =========================================================

if st.button(
    "Predict"
):

    input_df = pd.DataFrame(
        [input_values]
    )


    prediction = model.predict(
        input_df
    )[0]


    st.success(
        f"Predicted {target_column}: "
        f"{prediction:,.2f}"
    )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Regression Analysis Project | "
    "Multiple Linear Regression | "
    "50 Startups & Boston Housing"
)