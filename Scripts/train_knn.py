# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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

# Define the parameter grid for k (number of neighbors)
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Create a KNN classifier with improved settings
knn_classifier = KNeighborsClassifier(weights='distance', algorithm='auto', metric='manhattan')

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_k = grid_search.best_params_['n_neighbors']

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Create a new KNN classifier with the best k and improved settings
optimal_knn_classifier = KNeighborsClassifier(n_neighbors=best_k, weights='distance', algorithm='auto', metric='manhattan')

# Train the classifier on the training data
optimal_knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the validation set
optimal_knn_predictions = optimal_knn_classifier.predict(X_val_scaled)

# Evaluate the accuracy and F1 score of the optimized KNN classifier on the validation set
optimal_knn_accuracy = accuracy_score(y_val, optimal_knn_predictions)
optimal_knn_f1 = f1_score(y_val, optimal_knn_predictions, average='weighted')

print("Optimal KNN Accuracy on Validation Set:", optimal_knn_accuracy)
print("Optimal KNN F1 Score on Validation Set:", optimal_knn_f1)

# Save both the trained KNN model and the scaler to files using joblib
joblib.dump(optimal_knn_classifier, 'Kek/results/models/knn_model.pkl')
joblib.dump(scaler, 'Kek/results/scalers/knn_scaler.pkl')
