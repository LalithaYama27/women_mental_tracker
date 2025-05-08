# ensemble_cross_validation.py

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load a sample dataset (you can replace this with your own dataset)
data = load_iris()
X = data.data
y = data.target

# Define individual models
model_1 = LogisticRegression(max_iter=200)
model_2 = DecisionTreeClassifier(random_state=42)
model_3 = SVC(probability=True, random_state=42)

# Combine models into an ensemble using VotingClassifier
voting_model = VotingClassifier(estimators=[
    ('logreg', model_1),
    ('dt', model_2),
    ('svc', model_3)
], voting='soft')

# Perform 5-fold cross-validation on the ensemble model
cv_scores = cross_val_score(voting_model, X, y, cv=5)

# Output the results
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average Cross-Validation Score: {cv_scores.mean():.2f}")

