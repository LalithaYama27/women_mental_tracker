from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a sample dataset (you can replace this with your dataset)
data = load_iris()
X = data.data
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual models
model_1 = LogisticRegression(max_iter=1000)
model_2 = DecisionTreeClassifier()
model_3 = SVC()

# Create an ensemble voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('logreg', model_1),
    ('dt', model_2),
    ('svc', model_3)
], voting='hard')

# Define parameter grid for each model (you can tune parameters specific to each model)
param_grid = {
    'logreg__C': [0.1, 1, 10],
    'logreg__penalty': ['l2'],
    'dt__max_depth': [5, 10, 15],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

# Apply GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=ensemble_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters and performance
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
