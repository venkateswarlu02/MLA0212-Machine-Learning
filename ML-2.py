import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)  # For reproducibility
n_samples = 1000

income = np.random.randint(30000, 100000, n_samples)
age = np.random.randint(18, 70, n_samples)
loan_amount = np.random.randint(1000, 50000, n_samples)
credit_score = np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples, p=[0.2, 0.3, 0.3, 0.2])

label_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
credit_score = np.vectorize(label_map.get)(credit_score)

X = np.column_stack((income, age, loan_amount))
y = credit_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Poor', 'Fair', 'Good', 'Excellent'], zero_division=0)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
