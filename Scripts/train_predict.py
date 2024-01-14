import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import load

# Load the test dataset
df_test = pd.read_csv('data/mfcc_features_test.csv')

# Extract features from the test dataset
X_test = df_test.iloc[:, 2:].values

# Assuming you have already loaded your training dataset 'data/mfcc_features_train.csv'
df_train = pd.read_csv('data/mfcc_features_train.csv')

# Extract features and labels
X_train = df_train.iloc[:, 2:15].values  # Assuming columns 'MFCC_1' to 'MFCC_13' are your features
y_train = df_train['ClassID'].values

# Reshape the input features for SVM
X_train_svm = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test_svm = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Load the saved SVM model
loaded_svm_model = load('results/models/svm_model.pkl')
predictions_svm = loaded_svm_model.predict(X_test_svm)

# Rescale the input features for Naive Bayes
scaler_nb = StandardScaler()
X_train_scaled_nb = scaler_nb.fit_transform(X_train)
X_test_scaled_nb = scaler_nb.transform(X_test)

# Load the saved Naive Bayes model
loaded_nb_model = load('results/models/naive_bayes_model.pkl')
predictions_nb = loaded_nb_model.predict(X_test_scaled_nb)

# Reshape the input features for MLP
X_train_mlp = X_train
X_test_mlp = X_test


# Create and fit a scaler using the training data
scaler_mlp = StandardScaler()
X_train_scaled_mlp = scaler_mlp.fit_transform(X_train)
# Scale the test data using the scaler directly
X_test_mlp_scaled = scaler_mlp.transform(X_test_mlp)

# Load the saved MLP model
loaded_mlp_model = load('results/models/mlp_model.pkl')
predictions_mlp = loaded_mlp_model.predict(X_test_mlp_scaled)

# Load the saved ensemble model
loaded_ensemble_model = load('results/models/ensemble.pkl')

# Combine predictions from SVM, Naive Bayes, MLP, and the ensemble using argmax
ensemble_predictions = np.argmax(predictions_svm + predictions_nb + predictions_mlp +
                                loaded_ensemble_model.predict(X_test), axis=1)

# Create a new dataframe with 'ID' and 'PredictedLabel' columns for the ensemble
df_result_ensemble = pd.DataFrame({'ID': df_test['ID'], 'PredictedLabel': ensemble_predictions})

# Save the results of the ensemble to a new CSV file
df_result_ensemble.to_csv('results/submissions/submission_ensemble.csv', index=False)
