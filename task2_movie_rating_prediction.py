# Movie Rating Prediction using Python
# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
print("Loading dataset...")

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("\nFirst 5 rows:")
print(df.head())


# Display column names
print("\nColumns in dataset:")
print(df.columns)


# Remove rows where Rating is missing
df = df.dropna(subset=["Rating"])


# Select useful columns
features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

X = df[features]
y = df["Rating"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Handle missing values and convert text to numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            ),
            features
        )
    ]
)


# Create machine learning model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# Train model
print("\nTraining model...")
model.fit(X_train, y_train)


# Predict ratings
y_pred = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("------------------")
print("Mean Absolute Error :", round(mae, 2))
print("R2 Score            :", round(r2, 2))


# Example prediction
new_movie = pd.DataFrame({
    "Genre": ["Action"],
    "Director": ["S. S. Rajamouli"],
    "Actor 1": ["Prabhas"],
    "Actor 2": ["Rana Daggubati"],
    "Actor 3": ["Anushka Shetty"]
})

predicted_rating = model.predict(new_movie)

print("\nPredicted Movie Rating:", round(predicted_rating[0], 2))