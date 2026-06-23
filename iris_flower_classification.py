# Iris Flower Classification
# Author: Akhil
# Task 3 - CodSoft Data Science Internship

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_dataset():
    """Load the Iris dataset"""
    print("Loading dataset...\n")

    df = pd.read_csv("Iris.csv")

    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())

    return df


def prepare_data(df):
    """Prepare features and target variable"""

    X = df.drop("species", axis=1)
    y = df["species"]

    return X, y


def train_model(X_train, y_train):
    """Train Random Forest Classifier"""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Performance")
    print("-" * 25)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def predict_new_flower(model):
    """Predict species for a sample flower"""

    sample_flower = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2]
    })

    prediction = model.predict(sample_flower)

    print("\nSample Prediction")
    print("-" * 25)
    print("Predicted Species:", prediction[0])


def main():

    # Load dataset
    df = load_dataset()

    # Prepare data
    X, y = prepare_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Example prediction
    predict_new_flower(model)


if __name__ == "__main__":
    main()