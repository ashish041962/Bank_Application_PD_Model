import pickle
import pandas as pd


# Load simplified PD model
with open("models/simple_pd_model_v1.pkl", "rb") as file:
    simple_model = pickle.load(file)


# Load simplified scaler
with open("models/simple_scaler_v1.pkl", "rb") as file:
    simple_scaler = pickle.load(file)


# Load final internal model columns
with open("models/simple_features_v1.pkl", "rb") as file:
    simple_features = pickle.load(file)


# Load numeric columns
with open("models/simple_numeric_cols_v1.pkl", "rb") as file:
    simple_numeric_cols = pickle.load(file)


print("Simplified PD model loaded successfully")
print("Internal model columns:", len(simple_features))
print("Numeric columns:", len(simple_numeric_cols))

def predict_simple_pd(
    requested_amount,
    tenure_months,
    monthly_income,
    monthly_obligation,
    bureau_score,
    employment_type,
    employment_tenure,
    delinquencies_12m
):

    # Calculate FOIR automatically
    if monthly_income > 0:
        foir = monthly_obligation / monthly_income
    else:
        foir = 0.0

    # Create applicant data
    applicant = pd.DataFrame([{
        "REQUESTED_AMOUNT": requested_amount,
        "TENURE_MONTHS": tenure_months,
        "MONTHLY_INCOME": monthly_income,
        "TOTAL_MONTHLY_OBLIGATION": monthly_obligation,
        "BUREAU_SCORE": bureau_score,
        "EMPLOYMENT_TENURE_MONTHS": employment_tenure,
        "DELINQUENCIES_12M": delinquencies_12m,
        "FOIR_CALCULATED": foir
    }])

    # Scale numeric variables
    applicant[simple_numeric_cols] = (
        simple_scaler.transform(
            applicant[simple_numeric_cols]
        )
    )

    # Create employment dummy columns
    employment_column = (
        f"EMPLOYMENT_TYPE_{employment_type}"
    )

    applicant[employment_column] = 1

    # Match the exact 11 columns used by the model
    applicant = applicant.reindex(
        columns=simple_features,
        fill_value=0
    )

    # Calculate Probability of Default
    pd_probability = simple_model.predict_proba(
        applicant
    )[:, 1]

    return pd_probability[0]