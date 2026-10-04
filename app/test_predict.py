from predict_simple import predict_simple_pd


pd_result = predict_simple_pd(
    requested_amount=422000,
    tenure_months=36,
    monthly_income=143139,
    monthly_obligation=16376,
    bureau_score=740,
    employment_type="SALARIED",
    employment_tenure=191,
    delinquencies_12m=0
)


print(
    "Simplified Model PD:",
    round(pd_result * 100, 2),
    "%"
)