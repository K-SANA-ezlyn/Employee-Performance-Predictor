import pandas as pd
import numpy as np
import os

# Fix random values for same output every time
np.random.seed(42)


def create_employee_dataset(num_records=1000):
    """
    Creates synthetic employee dataset
    and saves it as employee_data.csv
    """

    departments = [
        "IT",
        "HR",
        "Sales",
        "Finance",
        "Marketing"
    ]

    data = {
        "Age": np.random.randint(22, 60, num_records),

        "Experience": np.random.randint(1, 20, num_records),

        "Department": np.random.choice(
            departments,
            num_records
        ),

        "Salary": np.random.randint(
            25000,
            120000,
            num_records
        ),

        "Training_Hours": np.random.randint(
            5,
            100,
            num_records
        ),

        "Attendance": np.random.randint(
            60,
            100,
            num_records
        ),

        "Projects_Completed": np.random.randint(
            1,
            15,
            num_records
        ),

        "Overtime_Hours": np.random.randint(
            0,
            50,
            num_records
        ),

        "Manager_Feedback": np.random.randint(
            1,
            10,
            num_records
        ),

        "Work_Life_Balance": np.random.randint(
            1,
            10,
            num_records
        )
    }

    df = pd.DataFrame(data)

    # Performance score logic
    score = (
        df["Training_Hours"] * 0.15
        + df["Attendance"] * 0.25
        + df["Projects_Completed"] * 2
        + df["Manager_Feedback"] * 5
        - df["Overtime_Hours"] * 0.2
        + df["Work_Life_Balance"] * 3
    )

    # Classify performance
    conditions = [
        score >= 60,
        (score >= 40) & (score < 60),
        score < 40
    ]

    choices = [
        "High",
        "Medium",
        "Low"
    ]

    df["Performance_Rating"] = np.select(
        conditions,
        choices,
        default="Medium"
    )

    # Create data folder if missing
    os.makedirs("data", exist_ok=True)

    # Save CSV
    df.to_csv(
        "data/employee_data.csv",
        index=False
    )

    print("Dataset created successfully!")
    print("Saved at: data/employee_data.csv")
    print(df.head())


if __name__ == "__main__":
    create_employee_dataset()

