from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import joblib

# Load the augmented dataset for prediction
new_data = pd.read_csv('data/mfcc_features_augmented_train.csv')

# Skip the first two columns (assuming they are not needed as features)
X = new_data.iloc[:, 2:].drop('ClassID', axis=1)  # Drop 'ClassID' column
y_true = new_data['ClassID']

# Load the trained models and scalers
knn_model = joblib.load('results/models/knn_model_augmented.pkl')
svm_model = joblib.load('results/models/svm_model_augmented.pkl')
random_forest_model = joblib.load('results/models/random_forest_model_augmented.pkl')
xgboost_model = joblib.load('results/models/xgboost_model_augmented.pkl')
mlp_model = joblib.load('results/models/mlp_model_augmented.pkl')

knn_scaler = None  # Assuming no scaler is used for KNN
svm_scaler = joblib.load('results/scalers/svm_scaler_augmented.pkl')  # Scaler used for SVM
xgboost_scaler = None  # Assuming no scaler is used for XGBoost
mlp_scaler = joblib.load('results/scalers/mlp_scaler_augmented.pkl')  # Scaler used for MLP

# Scale the input data
X_svm_scaled = svm_scaler.transform(X)
X_mlp_scaled = mlp_scaler.transform(X)

# List of models and their corresponding names
models = {
    'KNN': knn_model,
    'SVM': svm_model,
    'RandomForest': random_forest_model,
    'XGBoost': xgboost_model,
    'MLP': mlp_model,
}

# Convert model predictions to numpy arrays
model_predictions = {
    'KNN': knn_model.predict(X) if knn_scaler is None else knn_model.predict(X),
    'SVM': svm_model.predict(X_svm_scaled),
    'RandomForest': random_forest_model.predict(X),
    'XGBoost': xgboost_model.predict(X) if xgboost_scaler is None else xgboost_model.predict(X),
    'MLP': mlp_model.predict(X_mlp_scaled),
}

# Function to get majority vote prediction
def majority_vote(predictions):
    return np.argmax(np.bincount(predictions))

# Function to calculate accuracy and F1 score
def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # Specify average parameter for multi-class classification
    return accuracy, f1

# List to store results
results = []

# Iterate through all combinations of models with sizes 1 to 4
for r in range(1, 5):  # Change 4 to the desired number of models in the combination
    for model_combination in combinations(models.keys(), r):
        # Get predictions for the current combination
        current_predictions = [model_predictions[model] for model in model_combination]
        
        # Get majority vote prediction
        final_prediction = np.apply_along_axis(majority_vote, axis=0, arr=np.vstack(current_predictions))
        
        # Evaluate the final prediction
        accuracy, f1 = evaluate_predictions(y_true, final_prediction)
        
        # Append results to the list
        results.append({'Models': model_combination, 'Accuracy': accuracy, 'F1 Score': f1})

# Sort the results based on accuracy and print the top 4
top_accuracy_models = sorted(results, key=lambda x: x['Accuracy'], reverse=True)[:4]
print("Top 4 Models based on Accuracy:")
for result in top_accuracy_models:
    print(f"Models: {result['Models']}, Accuracy: {result['Accuracy']}, F1 Score: {result['F1 Score']}")

# Sort the results based on F1 score and print the top 4
top_f1_models = sorted(results, key=lambda x: x['F1 Score'], reverse=True)[:4]
print("\nTop 4 Models based on F1 Score:")
for result in top_f1_models:
    print(f"Models: {result['Models']}, Accuracy: {result['Accuracy']}, F1 Score: {result['F1 Score']}")

# Load the test dataset for prediction
test_data = pd.read_csv('data/mfcc_features_test.csv')

# Skip the first column (assuming it is not needed as a feature)
X_test = test_data.iloc[:, 2:]

# Scale the input test data
X_test_svm_scaled = svm_scaler.transform(X_test)
X_test_mlp_scaled = mlp_scaler.transform(X_test)

# List to store results for test predictions
test_results = []

# Specify the directory path for saving CSV files
save_directory = 'results/submissions/'

# Iterate through the top 4 combinations based on accuracy
for idx, top_models in enumerate(top_accuracy_models):
    # Get the models in the current combination
    current_models = [models[model_name] for model_name in top_models['Models']]
    
    # Get predictions for the test dataset
    test_predictions = []
    for model in current_models:
        if model == svm_model:
            test_predictions.append(model.predict(X_test_svm_scaled))
        elif model == mlp_model:
            test_predictions.append(model.predict(X_test_mlp_scaled))
        else:
            test_predictions.append(model.predict(X_test))
    
    # Get majority vote prediction for the test dataset
    final_test_prediction = np.apply_along_axis(majority_vote, axis=0, arr=np.vstack(test_predictions))
    
    # Save the predictions to a CSV file
    combination_name = f"{', '.join(top_models['Models'])}_Combination_{idx + 1}_Accuracy"
    prediction_df = pd.DataFrame({'id': test_data.index, 'label': final_test_prediction})
    prediction_df.to_csv(f'{save_directory}{combination_name}_predictions.csv', index=False)
    
    # Append results to the test_results list
    test_results.append({'Combination Name': combination_name, 'Models': top_models['Models'], 'Prediction CSV': f'{combination_name}_predictions.csv'})

# Iterate through the top 4 combinations based on F1 score
for idx, top_models in enumerate(top_f1_models):
    # Get the models in the current combination
    current_models = [models[model_name] for model_name in top_models['Models']]
    
    # Get predictions for the test dataset
    test_predictions = []
    for model in current_models:
        if model == svm_model:
            test_predictions.append(model.predict(X_test_svm_scaled))
        elif model == mlp_model:
            test_predictions.append(model.predict(X_test_mlp_scaled))
        else:
            test_predictions.append(model.predict(X_test))
    
    # Get majority vote prediction for the test dataset
    final_test_prediction = np.apply_along_axis(majority_vote, axis=0, arr=np.vstack(test_predictions))
    
    # Save the predictions to a CSV file
    combination_name = f"{', '.join(top_models['Models'])}_Combination_{idx + 1}_F1_Score"
    prediction_df = pd.DataFrame({'id': test_data.index, 'label': final_test_prediction})
    prediction_df.to_csv(f'{save_directory}{combination_name}_predictions.csv', index=False)
    
    # Append results to the test_results list
    test_results.append({'Combination Name': combination_name, 'Models': top_models['Models'], 'Prediction CSV': f'{combination_name}_predictions.csv'})

# Print the results for the test predictions
print("\nTest Predictions:")
for result in test_results:
    combination_name = f"{', '.join(result['Models'])}_{result['Combination Name']}"
    print(f"Combination Name: {combination_name}, Models: {result['Models']}, Prediction CSV: {save_directory}{result['Prediction CSV']}")
    #print(f"Combination Name: {result['Combination Name']}, Models: {result['Models']}, Prediction CSV: {result['Prediction CSV']}")
