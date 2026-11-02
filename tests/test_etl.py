import pandas as pd

from src.etl import TARGET, transform


def sample():
    return pd.DataFrame({
        "credit_amount": [1000, 2400, 1000],
        "duration": [10, 12, 10],
        "purpose": ["car", "radio/tv", "car"],
        "class": ["good", "bad", "good"],
    })


def test_transform_target_and_feature():
    out = transform(sample())
    assert TARGET in out.columns and "class" not in out.columns
    assert len(out) == 2  # duplicate dropped
    assert set(out[TARGET]) == {0, 1}
    assert out["credit_per_month"].iloc[0] == 100.0
