import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('./salary_dataset.csv')

X = np.array(data["YearsExperience"]).reshape(-1, 1)
y = np.array(data["Salary"])

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

print("ชุดฝึก (X_train):", X_train)
print("ชุดฝึก (y_train):", y_train)
print("ชุดทดสอบ (X_test):", X_test)
print("ชุดทดสอบ (y_test):", y_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

experience_year = float(input(""))

new_X = np.array([experience_year]).reshape(-1, 1)
salary_predict = model.predict(new_X)
print(f"Salary: {salary_predict}")
