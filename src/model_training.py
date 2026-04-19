from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
import os


def train_model():
    """
    Trains ML model
    and saves model file
    """

    df = preprocess_data()

    X = df.drop(
        "Performance_Rating",
        axis=1
    )

    y = df["Performance_Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/employee_model.pkl"
    )

    print("\nModel Training Complete")
    print(f"\nAccuracy: {accuracy:.2f}")

    print("\nClassification Report:\n")
    print(report)

    print("\nConfusion Matrix:\n")
    print(matrix)


if __name__ == "__main__":
    train_model()
