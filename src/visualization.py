import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_visualizations():
    """
    Generate graphs and save them
    """

    df = pd.read_csv(
        "data/employee_data.csv"
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # Graph 1: Performance Rating Count
    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="Performance_Rating",
        data=df
    )

    plt.title("Performance Rating Distribution")

    plt.savefig(
        "outputs/performance_distribution.png"
    )

    plt.close()

    # Graph 2: Attendance vs Performance
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x="Performance_Rating",
        y="Attendance",
        data=df
    )

    plt.title("Attendance vs Performance")

    plt.savefig(
        "outputs/attendance_vs_performance.png"
    )

    plt.close()

    print("Visualizations created successfully!")
    print("Saved inside outputs folder")


if __name__ == "__main__":
    create_visualizations()
