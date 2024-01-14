# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(rf_classifier, '../results/models/random_forest_model.pkl')

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate the F1 score
f1 = f1_score(y_test, predictions, average='weighted')  # Specify average parameter
print("F1 Score:", f1)
