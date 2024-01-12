import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the test data
test_data_path = "data/mfcc_features_test.csv"  # Update the path as needed
test_data = pd.read_csv(test_data_path)

# Load the pre-trained models
mlp_model = joblib.load('results/models/mlp_model.pkl')
svm_model = joblib.load('results/models/svm_model.pkl')
nb_model = joblib.load('results/models/naive_bayes_model.pkl')
ensemble_model = joblib.load('results/models/ensemble.pkl')

# Extract features from the test data
X_test = test_data.iloc[:, 2:].values  # Assuming features start from the 3rd column

# Feature scaling for the individual models
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Extract features from the test set using individual models
X_test_mlp = mlp_model.predict(X_test_scaled)
X_test_svm = svm_model.predict(X_test_scaled)
X_test_nb = nb_model.predict(X_test_scaled)

# Create a DataFrame with the predictions from individual models for the test set
ensemble_test_data = pd.DataFrame({
    'MLP_Prediction': X_test_mlp,
    'SVM_Prediction': X_test_svm,
    'NaiveBayes_Prediction': X_test_nb
})

# Make predictions on the test set using the ensemble model
ensemble_test_predictions = ensemble_model.predict(ensemble_test_data)

# Create a submission DataFrame
submission_df = pd.DataFrame({
    "ID": test_data["ID"],
    "ClassID": ensemble_test_predictions
})

# Save the submission DataFrame to a CSV file
submission_path = "results/submissions/submission.csv"  # Update the path as needed
submission_df.to_csv(submission_path, index=False)

print("Submission CSV generated successfully.")
