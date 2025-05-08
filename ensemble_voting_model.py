# 1. Import Necessary Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 2. Load Dataset (Replace with your own dataset)
# For demonstration, using the iris dataset
X, y = load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Models (Train these from scratch)
model_1 = LogisticRegression(max_iter=200)  # Logistic Regression model
model_2 = DecisionTreeClassifier(random_state=42)  # Decision Tree model
model_3 = SVC(probability=True, random_state=42)  # SVC model (with probability=True for soft voting)

# 4. Train the Individual Models
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)

# 5. Set Up the Voting Classifier (Hard or Soft Voting)
voting_model = VotingClassifier(estimators=[
    ('lr', model_1),
    ('dt', model_2),
    ('svc', model_3)
], voting='hard')  # Change to 'soft' for soft voting

# 6. Train the Ensemble Model
voting_model.fit(X_train, y_train)

# 7. Evaluate the Ensemble Model
y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Model Accuracy: {accuracy * 100:.2f}%')

# Optionally, Save the Ensemble Model for Future Use
import joblib
joblib.dump(voting_model, 'ensemble_voting_model.pkl')
