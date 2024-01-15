# Import necessary libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load the stratified training and validation data
train_data = pd.read_csv('Kek\Data\stratified_training_set.csv')
val_data = pd.read_csv('Kek\Data\stratified_validation_set.csv')

# Assuming the features start from the 3rd column
X_train = train_data.iloc[:, 2:-1]
y_train = train_data['ClassID']

X_val = val_data.iloc[:, 2:-1]
y_val = val_data['ClassID']

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_scaled, y_train)

# Training Accuracy
train_predictions = xgb_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Validation Accuracy and F1 Score
xgb_predictions = xgb_classifier.predict(X_val_scaled)
xgb_accuracy = accuracy_score(y_val, xgb_predictions)
xgb_f1 = f1_score(y_val, xgb_predictions, average='weighted')

print("XGBoost Validation Accuracy:", xgb_accuracy)
print("XGBoost Validation F1 Score:", xgb_f1)

# Save the trained XGBoost model and scaler to files using joblib
joblib.dump(xgb_classifier, 'Kek/results/models/xgboost_model_with_scaling.pkl')
joblib.dump(scaler, 'Kek/results/scalers/xgboost_model_with_scaling_scaler_xgb.pkl')
