import pandas as pd


def run_eda():
    """
    Basic Exploratory Data Analysis
    """

    df = pd.read_csv(
        "data/employee_data.csv"
    )

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())


if __name__ == "__main__":
    run_eda()
