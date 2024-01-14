import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import joblib

# Load the training data
train_data = pd.read_csv('data/mfcc_features_train.csv')

# Split the data into features (X) and labels (y)
X = train_data.iloc[:, 2:-1]  # Assuming MFCC features start from the 3rd column and the last column is the ClassID
y = train_data['ClassID']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize Gaussian Naive Bayes model
nb_model = GaussianNB()

# Perform a basic grid search for parameter tuning
param_grid = {
    'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}

cv = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(nb_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_nb_model = grid_search.best_estimator_

best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train the best model on the entire training set
best_nb_model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred = best_nb_model.predict(X_val_scaled)

# Evaluate the best model on the validation set
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy on validation set: {accuracy}')

# Save the best trained model
joblib.dump(best_nb_model, 'results/models/naive_bayes_model.pkl')
