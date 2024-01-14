from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load preprocessed data
data = pd.read_csv("../data/mfcc_features_augmented_train.csv")
X = data.iloc[:, 2:-1]
y = data["ClassID"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the parameter grid for grid search
param_grid = {
    'C': [1000],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf'],
    'degree': [2, 3, 4],
    'probability': [True]
}

# Create an SVM classifier
svm_clf = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from grid search
best_params = grid_search.best_params_

# Train the final SVM model with the best parameters
final_svm_model = SVC(**best_params)
final_svm_model.fit(X_train_scaled, y_train)

# Save the trained SVM model and the scaler
joblib.dump(final_svm_model, "../results/models/svm_model_augmented.pkl")
joblib.dump(scaler, "../results/scalers/svm_scaler_augmented.pkl")

# Evaluate the model on the validation set
y_val_pred = final_svm_model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Print the best parameters
print("Best Parameters:", best_params)
