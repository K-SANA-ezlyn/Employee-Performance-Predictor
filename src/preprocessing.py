import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data():
    """
    Reads dataset and converts
    categorical values into numbers
    """

    df = pd.read_csv(
        "data/employee_data.csv"
    )

    encoder = LabelEncoder()

    # Convert Department
    df["Department"] = encoder.fit_transform(
        df["Department"]
    )

    # Convert Target Column
    df["Performance_Rating"] = encoder.fit_transform(
        df["Performance_Rating"]
    )

    return df
