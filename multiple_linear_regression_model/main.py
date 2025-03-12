import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("./student_performance.csv")

print(data)
print(data.head())
print(data.info())
print(data.describe())

y = np.array(data["Performance Index"])
X = np.array(data.drop(["Performance Index"], axis=1))

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

print("ชุดฝึก (X_train):", X_train)
print("ชุดฝึก (y_train):", y_train)
print("ชุดทดสอบ (X_test):", X_test)
print("ชุดทดสอบ (y_test):", y_test)

model = LinearRegression()

model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

y_predict = model.predict(X_test)

mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")