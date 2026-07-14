"""
Feature engineering transformer used both in the training notebook and the
Streamlit app, so the exact same logic runs at train time and inference time.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds clinically-motivated derived features to the raw 13 input columns.

    New columns:
        age_group        -- binned age decade (0-4)
        chol_per_age      -- cholesterol normalized by age
        hr_reserve         -- heart-rate reserve vs. age-predicted max (220 - age)
        risk_factor_count  -- count of binary risk indicators present
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data["age_group"] = pd.cut(
            data["age"], bins=[0, 40, 50, 60, 70, 120], labels=[0, 1, 2, 3, 4]
        ).astype(int)
        data["chol_per_age"] = data["chol"] / data["age"]
        data["hr_reserve"] = (220 - data["age"]) - data["thalach"]
        data["risk_factor_count"] = (
            (data["sex"] == 1).astype(int)
            + (data["fbs"] == 1).astype(int)
            + (data["exang"] == 1).astype(int)
            + (data["chol"] > 240).astype(int)
        )
        return data

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + ["age_group", "chol_per_age", "hr_reserve", "risk_factor_count"]
