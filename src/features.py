from sklearn.preprocessing import StandardScaler
from src.config import NUMERIC_COLS


def create_features(x_train, x_cv, x_test):
    """Create BMI feature for all datasets"""
    for df in [x_train, x_cv, x_test]:
        df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    return x_train, x_cv, x_test


def scale_features(x_train, x_cv, x_test):
    """Scale numeric features"""
    scaler = StandardScaler()

    x_train[NUMERIC_COLS] = scaler.fit_transform(x_train[NUMERIC_COLS])
    x_cv[NUMERIC_COLS] = scaler.transform(x_cv[NUMERIC_COLS])
    x_test[NUMERIC_COLS] = scaler.transform(x_test[NUMERIC_COLS])

    return x_train, x_cv, x_test, scaler
