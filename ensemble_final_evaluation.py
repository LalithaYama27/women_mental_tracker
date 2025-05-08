from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris

# Load a sample dataset (you can replace this with your dataset)
data = load_iris()
X = data.data
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual models
model_1 = LogisticRegression(max_iter=1000)
model_2 = DecisionTreeClassifier(max_depth=5)
model_3 = SVC(C=0.1, kernel='linear')

# Create an ensemble voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('logreg', model_1),
    ('dt', model_2),
    ('svc', model_3)
], voting='hard')

# Train all models
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred_logreg = model_1.predict(X_test)
y_pred_dt = model_2.predict(X_test)
y_pred_svc = model_3.predict(X_test)
y_pred_ensemble = ensemble_model.predict(X_test)

# Calculate accuracy
acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_svc = accuracy_score(y_test, y_pred_svc)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)

# Print the performance of each model
print(f"Logistic Regression Accuracy: {acc_logreg * 100:.2f}%")
print(f"Decision Tree Accuracy: {acc_dt * 100:.2f}%")
print(f"SVC Accuracy: {acc_svc * 100:.2f}%")
print(f"Ensemble Model Accuracy: {acc_ensemble * 100:.2f}%")
