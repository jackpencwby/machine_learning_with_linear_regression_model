import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([40, 50, 60, 70, 80, 90])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

print("ชุดฝึก (X_train):", X_train)
print("ชุดฝึก (y_train):", y_train)
print("ชุดทดสอบ (X_test):", X_test)
print("ชุดทดสอบ (y_test):", y_test)

# สร้างโมเดล Linear Regression
model = LinearRegression()

# ฝึกโมเดลด้วยชุดฝึก
model.fit(X_train, y_train)

# ทำนายผลลัพธ์ด้วยชุดทดสอบ
y_predict = model.predict(X_test)

# ประเมินผลโมเดล
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# ทำนายคะแนนสอบสำหรับจำนวนชั่วโมงเรียนใหม่
hour = float(input(""))

new_X = np.array([hour]).reshape(-1, 1)
score_predict = model.predict(new_X)
print(f"Predicted score for {hour} hours of study: {score_predict[0]}")