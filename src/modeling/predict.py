import pandas as pd
from src.config import NUMERIC_COLS


def predict_new_sample(model, scaler, sample_data):
    """Make prediction on new sample"""
    new_sample = pd.DataFrame([sample_data])

    # Create BMI feature
    new_sample["BMI"] = new_sample["Weight"] / (new_sample["Height"] / 100) ** 2

    # Scale features
    new_sample[NUMERIC_COLS] = scaler.transform(new_sample[NUMERIC_COLS])

    # Predict
    prediction = model.predict(new_sample)
    return prediction[0][0]
