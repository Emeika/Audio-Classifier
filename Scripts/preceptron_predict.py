import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import joblib

# Load the test data
test_data = pd.read_csv('data/mfcc_features_test.csv')

# Assuming your test data has the same format as the training data
X_test = test_data.iloc[:, 2:]

# Standardize the features (using the same scaler as during training)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Load the trained MLP model
mlp_model = joblib.load('results/models/mlp_model_extended.pkl')

# Make predictions on the test set
y_test_pred = mlp_model.predict(X_test_scaled)

# Create a DataFrame with the predicted labels
submission_df = pd.DataFrame({'ID': test_data['ID'], 'Label': y_test_pred})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('results/submissions/preceptron_predictions.csv', index=False)

# Display the first few rows of the submission DataFrame
print(submission_df.head())
