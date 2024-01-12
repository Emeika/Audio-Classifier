import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('data/mfcc_features_train.csv')
X = train_data.iloc[:, 2:-1]
y = train_data['ClassID']

# Split the data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained models
mlp_model = joblib.load('results/models/mlp_model.pkl')
svm_model = joblib.load('results/models/svm_model.pkl')
nb_model = joblib.load('results/models/naive_bayes_model.pkl')

# Feature scaling for the individual models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Extract features from the training set
X_train_mlp = mlp_model.predict(X_train_scaled)
X_train_svm = svm_model.predict(X_train_scaled)
X_train_nb = nb_model.predict(X_train_scaled)

# Create a DataFrame with the predictions from individual models
ensemble_train_data = pd.DataFrame({
    'MLP_Prediction': X_train_mlp,
    'SVM_Prediction': X_train_svm,
    'NaiveBayes_Prediction': X_train_nb
})

# Train a Random Forest classifier on the ensemble predictions
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(ensemble_train_data, y_train)

# Feature scaling for the validation set
X_val_scaled = scaler.transform(X_val)

# Extract features from the validation set
X_val_mlp = mlp_model.predict(X_val_scaled)
X_val_svm = svm_model.predict(X_val_scaled)
X_val_nb = nb_model.predict(X_val_scaled)

# Create a DataFrame with the predictions from individual models for validation
ensemble_val_data = pd.DataFrame({
    'MLP_Prediction': X_val_mlp,
    'SVM_Prediction': X_val_svm,
    'NaiveBayes_Prediction': X_val_nb
})

# Make predictions on the validation set using the ensemble
ensemble_val_predictions = rf_classifier.predict(ensemble_val_data)

# Evaluate the ensemble model on the validation set
accuracy_ensemble = accuracy_score(y_val, ensemble_val_predictions)
print(f'Accuracy of the Ensemble Model on Validation Set: {accuracy_ensemble}')

# Save the trained ensemble model
ensemble_model_path = 'results/models/ensemble.pkl'
joblib.dump(rf_classifier, ensemble_model_path)
print(f'Ensemble model saved at: {ensemble_model_path}')
