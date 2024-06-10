import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error

np.random.seed(42)  # For reproducibility
n_students = 100

math_marks = np.random.randint(50, 100, n_students)
english_marks = np.random.randint(50, 100, n_students)
science_marks = np.random.randint(50, 100, n_students)
social_studies_marks = np.random.randint(50, 100, n_students)

# Calculating total and average marks
total_marks = math_marks + english_marks + science_marks + social_studies_marks
average_marks = total_marks / 4

X = np.column_stack((math_marks, english_marks, science_marks, social_studies_marks))
y_total = total_marks
y_avg = average_marks

X_train, X_test, y_train_total, y_test_total = train_test_split(X, y_total, test_size=0.2, random_state=42)
_, _, y_train_avg, y_test_avg = train_test_split(X, y_avg, test_size=0.2, random_state=42)

total_model = DecisionTreeRegressor(random_state=42)
total_model.fit(X_train, y_train_total)

avg_model = DecisionTreeRegressor(random_state=42)
avg_model.fit(X_train, y_train_avg)

y_pred_total = total_model.predict(X_test)
y_pred_avg = avg_model.predict(X_test)

total_mse = mean_squared_error(y_test_total, y_pred_total)
avg_mse = mean_squared_error(y_test_avg, y_pred_avg)

print(f"Mean Squared Error for Total marks prediction: {total_mse}")
print(f"Mean Squared Error for Average marks prediction: {avg_mse}")


tree_text = export_text(total_model, feature_names=['Math', 'English', 'Science', 'Social_Studies'])
print("Decision Tree for Total Marks Prediction:")
print(tree_text)
