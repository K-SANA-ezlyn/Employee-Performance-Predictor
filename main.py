from src.data_generator import create_employee_dataset
from src.model_training import train_model
from src.predict import predict_employee_performance
from src.visualization import create_visualizations
from src.eda import run_eda


def main():
    print("====================================")
    print("Employee Performance Predictor")
    print("====================================\n")

    print("STEP 1 → Creating Dataset\n")
    create_employee_dataset()

    print("\nSTEP 2 → Running EDA\n")
    run_eda()

    print("\nSTEP 3 → Training Model\n")
    train_model()

    print("\nSTEP 4 → Making Prediction\n")
    predict_employee_performance()

    print("\nSTEP 5 → Creating Visualizations\n")
    create_visualizations()

    print("\nPROJECT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
