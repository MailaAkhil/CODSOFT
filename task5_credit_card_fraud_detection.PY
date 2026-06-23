import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
print("Loading dataset...\n")

df = pd.read_csv("creditcard.csv")

print("Original Dataset Shape:", df.shape)

# Use a smaller sample for faster training
df = df.sample(n=50000, random_state=42)

print("Sampled Dataset Shape:", df.shape)

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
print("\nTraining model...\n")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\nModel Performance")
print("-" * 30)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Example prediction
sample_transaction = X_test.iloc[[0]]

prediction = model.predict(sample_transaction)

print("\nSample Transaction Prediction")
print("-" * 30)

if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Genuine Transaction")