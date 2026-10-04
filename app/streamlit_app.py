import streamlit as st

from predict_simple import predict_simple_pd


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Bank PD Model",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("🏦 Bank Probability of Default Model")

st.write(
    "Enter applicant information to estimate "
    "the Probability of Default (PD)."
)

st.info(
    "Model: Logistic Regression | "
    "Output: Probability of Default"
)


# --------------------------------------------------
# PAGE LAYOUT
# --------------------------------------------------

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# --------------------------------------------------
# LEFT SIDE - APPLICANT INPUTS
# --------------------------------------------------

with left_col:

    st.subheader("Applicant Details")

    requested_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=500000.0,
        step=10000.0
    )

    tenure_months = st.number_input(
        "Loan Tenure (Months)",
        min_value=1,
        value=36,
        step=1
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=100000.0,
        step=5000.0
    )

    monthly_obligation = st.number_input(
        "Total Monthly Obligation",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    bureau_score = st.number_input(
        "Bureau Score",
        min_value=300,
        max_value=900,
        value=750,
        step=1
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "SALARIED",
            "SELF_EMPLOYED",
            "PROFESSIONAL"
        ]
    )

    employment_tenure = st.number_input(
        "Employment Tenure (Months)",
        min_value=0,
        value=36,
        step=1
    )

    delinquencies_12m = st.number_input(
        "Delinquencies in Last 12 Months",
        min_value=0,
        value=0,
        step=1
    )

    # FOIR is calculated automatically
    if monthly_income > 0:
        foir = monthly_obligation / monthly_income
    else:
        foir = 0.0

    st.write(
        "Calculated FOIR:",
        f"{foir:.2%}"
    )


# --------------------------------------------------
# RIGHT SIDE - PD RESULT
# --------------------------------------------------

with right_col:

    st.subheader("PD Result")

    st.write(
        "Complete the applicant details on the left "
        "and click Calculate PD."
    )

    calculate_button = st.button(
        "Calculate PD",
        type="primary",
        use_container_width=True
    )

    reset_button = st.button(
        "Reset",
        use_container_width=True
    )

    # Calculate PD
    if calculate_button:

        pd_result = predict_simple_pd(
            requested_amount=requested_amount,
            tenure_months=tenure_months,
            monthly_income=monthly_income,
            monthly_obligation=monthly_obligation,
            bureau_score=bureau_score,
            employment_type=employment_type,
            employment_tenure=employment_tenure,
            delinquencies_12m=delinquencies_12m
        )

        st.metric(
            label="Probability of Default",
            value=f"{pd_result:.2%}"
        )

    # Reset the application
    if reset_button:
        st.rerun()