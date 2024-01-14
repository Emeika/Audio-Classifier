# Import necessary libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import pandas as pd

# Assume you have your dataset in X (features) and y (target)
data = pd.read_csv('../Data/mfcc_features_train.csv')

# Skip the first two columns (assuming they are not needed as features)
X = data.iloc[:, 2:].drop('ClassID', axis=1)  # Drop 'ClassID' column
y = data['ClassID']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions, average='weighted')

print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost F1 Score:", xgb_f1)

# Save the trained XGBoost model to a file using joblib
joblib.dump(xgb_classifier, '../results/models/xgboost_model.pkl')
