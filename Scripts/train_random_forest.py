# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load the augmented training data
train_data = pd.read_csv('Kek\Data\stratified_training_set.csv')
val_data = pd.read_csv('Kek\Data\stratified_validation_set.csv')

# Split the data into features and labels
X_train = train_data.iloc[:, 2:-1]  # Assuming the features start from the 3rd column
y_train = train_data['ClassID']

X_val = val_data.iloc[:, 2:-1]  # Assuming the features start from the 3rd column
y_val = val_data['ClassID']

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train_scaled, y_train)

# Save the trained model and scaler to files using joblib
joblib.dump(rf_classifier, 'Kek/results/models/random_forest_model_with_scaling.pkl')
joblib.dump(scaler, 'Kek/results/scalers/random_forest_scaler_rf.pkl')

# Make predictions on the validation set
predictions = rf_classifier.predict(X_val_scaled)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_val, predictions)
print("Accuracy:", accuracy)

# Calculate the F1 score
f1 = f1_score(y_val, predictions, average='weighted')  # Specify average parameter
print("F1 Score:", f1)
