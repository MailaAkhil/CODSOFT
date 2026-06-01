import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("titanic-dataset.csv")

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)



pclass = int(input("Enter Pclass (1, 2, 3): "))
sex = input("Enter Sex (male/female): ")
age = float(input("Enter Age: "))
fare = float(input("Enter Fare: "))
sex = 0 if sex == "male" else 1
input_data = [[pclass, sex, age, fare]]
input_data = pd.DataFrame([[pclass, sex, age, fare]],
                           columns=['Pclass', 'Sex', 'Age', 'Fare'])
prediction = model.predict(input_data)
if prediction[0] == 1:
    print("👉 Survived")
else:
    print("👉 Did NOT survive")