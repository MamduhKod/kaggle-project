import pandas as pd
from src.config import RAW_DATA_PATH


def load_raw_data():
    """Load and perform basic preprocessing on raw data"""
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(columns=["id"])
    df = pd.get_dummies(df, columns=["Sex"], drop_first=True)

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def split_data(df):
    """Split data into train, cv, and test sets"""
    from sklearn.model_selection import train_test_split
    from src.config import RANDOM_STATE

    Y = df[TARGET_COL]
    X = df.drop(TARGET_COL, axis=1)

    x_train, x_, y_train, y_ = train_test_split(
        X, Y, test_size=0.40, random_state=RANDOM_STATE
    )
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_, y_, test_size=0.50, random_state=RANDOM_STATE
    )

    return x_train, x_cv, x_test, y_train, y_cv, y_test
