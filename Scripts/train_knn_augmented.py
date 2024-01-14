# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import pandas as pd

# Assume you have your dataset in X (features) and y (target)
data = pd.read_csv('../Data/mfcc_features_augmented_train.csv')

# Skip the first two columns (assuming they are not needed as features)
X = data.iloc[:, 2:].drop('ClassID', axis=1)  # Drop 'ClassID' column
y = data['ClassID']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for k (number of neighbors)
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Create a KNN classifier
knn_classifier = KNeighborsClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Get the best parameter
best_k = grid_search.best_params_['n_neighbors']

# Create a new KNN classifier with the best k
optimal_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)

# Train the classifier on the training data
optimal_knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
optimal_knn_predictions = optimal_knn_classifier.predict(X_test)

# Evaluate the accuracy and F1 score of the optimized KNN classifier
optimal_knn_accuracy = accuracy_score(y_test, optimal_knn_predictions)
optimal_knn_f1 = f1_score(y_test, optimal_knn_predictions, average='weighted')

print("Optimal KNN Accuracy:", optimal_knn_accuracy)
print("Optimal KNN F1 Score:", optimal_knn_f1)

# Save the trained KNN model to a file using joblib
joblib.dump(optimal_knn_classifier, '../results/models/knn_model_augmented.pkl')
