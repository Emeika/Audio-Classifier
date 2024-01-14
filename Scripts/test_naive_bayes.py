# code/predict_naive_bayes.py

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained Naive Bayes model
best_nb_model = joblib.load('../results/models/best_naive_bayes_model.pkl')

# Load the testing data
test_data = pd.read_csv('../data/mfcc_features_test.csv')

# Extract features from the testing data
X_test = test_data.iloc[:, 2:]

# Feature scaling using the same scaler used during training
scaler = MinMaxScaler()  # You can use the same scaler as during training
X_test_scaled = scaler.fit_transform(X_test)

# Make predictions on the testing data
y_test_pred = best_nb_model.predict(X_test_scaled)

# Add the predicted labels to the testing data
test_data['PredictedClassID'] = y_test_pred

# Save the submission file
submission = test_data[['ID', 'PredictedClassID']]
submission.to_csv('../results/submissions/submission_naive_bayes.csv', index=False)
