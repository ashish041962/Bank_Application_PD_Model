import pickle
import pandas as pd


# Load trained PD model
with open("models/pd_model_v1.pkl", "rb") as file:
    pd_model = pickle.load(file)


# Load numeric scaler
with open("models/scaler_v1.pkl", "rb") as file:
    scaler = pickle.load(file)


# Load final model features
with open("models/final_features_v1.pkl", "rb") as file:
    final_features = pickle.load(file)


# Load numeric column names
with open("models/final_numeric_cols_v1.pkl", "rb") as file:
    numeric_cols = pickle.load(file)


# Load categorical column names
with open("models/final_categorical_cols_v1.pkl", "rb") as file:
    categorical_cols = pickle.load(file)


print("PD model loaded successfully")
print("Number of model features:", len(final_features))
print("Number of numeric features:", len(numeric_cols))
print("Number of categorical features:", len(categorical_cols))

def predict_pd(applicant):

    data = applicant.copy()

    # Handle collateral missing values
    collateral_numeric = [
        "COLLATERAL_VALUE",
        "LTV",
        "PROPERTY_AGE_YEARS"
    ]

    data[collateral_numeric] = (
        data[collateral_numeric].fillna(0)
    )

    data["PROPERTY_TYPE"] = (
        data["PROPERTY_TYPE"].fillna("NOT_APPLICABLE")
    )

    data["PROPERTY_CITY"] = (
        data["PROPERTY_CITY"].fillna("NOT_APPLICABLE")
    )

    # Create the 127 model input features
    model_input = pd.DataFrame(
        0.0,
        index=data.index,
        columns=final_features
    )

    # Scale numeric variables
    scaled_numeric = scaler.transform(
        data[numeric_cols]
    )

    model_input[numeric_cols] = scaled_numeric

    # Convert categorical values into dummy variables
    for col in categorical_cols:

        value = data.iloc[0][col]

        dummy_col = f"{col}_{value}"

        if dummy_col in final_features:
            model_input[dummy_col] = 1.0

    # Calculate Probability of Default
    pd_probability = pd_model.predict_proba(
        model_input
    )[:, 1]

    return pd_probability[0]