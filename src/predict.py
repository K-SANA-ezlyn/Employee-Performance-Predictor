import joblib
import pandas as pd


def predict_employee_performance():
    """
    Predict performance for a new employee
    """

    # Load trained model
    model = joblib.load(
        "models/employee_model.pkl"
    )

    # Example new employee data
    new_employee = pd.DataFrame([{
        "Age": 30,
        "Experience": 5,
        "Department": 2,  # Encoded value
        "Salary": 55000,
        "Training_Hours": 40,
        "Attendance": 92,
        "Projects_Completed": 7,
        "Overtime_Hours": 10,
        "Manager_Feedback": 8,
        "Work_Life_Balance": 7
    }])

    prediction = model.predict(new_employee)

    label_map = {
        0: "High",
        1: "Low",
        2: "Medium"
    }

    result = label_map.get(
        prediction[0],
        "Unknown"
    )

    print("\nPrediction Result")
    print("------------------")
    print(f"Predicted Performance: {result}")


if __name__ == "__main__":
    predict_employee_performance()
