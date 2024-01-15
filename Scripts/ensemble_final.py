from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import joblib

train_data = pd.read_csv('Kek\Data\stratified_training_set.csv')
val_data = pd.read_csv('Kek\Data\stratified_validation_set.csv')

# Split the data into features and labels
X_train = train_data.iloc[:, 2:-1]  # Assuming the features start from the 3rd column
y_train = train_data['ClassID']

X_val = val_data.iloc[:, 2:-1]  # Assuming the features start from the 3rd column
y_val = val_data['ClassID']

# Load the trained models and scalers
knn_model = joblib.load('Kek/results/models/knn_model.pkl')
svm_model = joblib.load('Kek/results/models/svm_model.pkl')
random_forest_model = joblib.load('Kek/results/models/random_forest_model_with_scaling.pkl')
xgboost_model = joblib.load('Kek/results/models/xgboost_model_with_scaling.pkl')
mlp_model = joblib.load('Kek/results/models/mlp_model_.pkl')

knn_scaler = joblib.load('Kek/results/scalers/knn_scaler.pkl')
xgboost_scaler = joblib.load('Kek/results/scalers/xgboost_model_with_scaling_scaler_xgb.pkl')
random_forest_scaler = joblib.load('Kek/results/scalers/random_forest_scaler_rf.pkl')
mlp_scaler = joblib.load('Kek/results/scalers/mlp_scaler.pkl')
svm_scaler = joblib.load('Kek/results/scalers/svm_scaler.pkl')

# Scale the input data
X_svm_scaled = svm_scaler.transform(X_val)
X_mlp_scaled = mlp_scaler.transform(X_val)
X_knn_scaled = knn_scaler.transform(X_val)
X_xgboost_scaled = xgboost_scaler.transform(X_val)

# List of models and their corresponding names
models = {
    'KNN': knn_model,
    'SVM': svm_model,
    'RandomForest': random_forest_model,
    'XGBoost': xgboost_model,
    'MLP': mlp_model,
}

scalers = {
    'KNN': knn_scaler,
    'SVM': svm_scaler,
    'RandomForest': random_forest_scaler,
    'XGBoost': xgboost_scaler,
    'MLP': mlp_scaler,
}

# Convert model predictions to numpy arrays
model_predictions = {
    'KNN': models['KNN'].predict(X_knn_scaled),
    'SVM': models['SVM'].predict(X_svm_scaled),
    'RandomForest': models['RandomForest'].predict(X_val),
    'XGBoost': models['XGBoost'].predict(X_xgboost_scaled),
    'MLP': models['MLP'].predict(X_mlp_scaled),
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
        accuracy, f1 = evaluate_predictions(y_val, final_prediction)
        
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
test_data = pd.read_csv('Kek\Data\mfcc_features_test.csv')

# Skip the first column (assuming it is not needed as a feature)
X_test = test_data.iloc[:, 2:]

# Scale the input test data
X_test_svm_scaled = svm_scaler.transform(X_test)
X_test_mlp_scaled = mlp_scaler.transform(X_test)

# List to store results for test predictions
test_results = []

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
    combination_name = f"Combination_{idx + 1}_Accuracy"
    prediction_df = pd.DataFrame({'id': test_data.index, 'label': final_test_prediction})
    prediction_df.to_csv(f'{combination_name}_predictions.csv', index=False)
    
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
    combination_name = f"Combination_{idx + 1}_F1_Score"
    prediction_df = pd.DataFrame({'id': test_data.index, 'label': final_test_prediction})
    prediction_df.to_csv(f'{combination_name}_predictions.csv', index=False)
    
    # Append results to the test_results list
    test_results.append({'Combination Name': combination_name, 'Models': top_models['Models'], 'Prediction CSV': f'{combination_name}_predictions.csv'})

# Print the results for the test predictions
print("\nTest Predictions:")
for result in test_results:
    print(f"Combination Name: {result['Combination Name']}, Models: {result['Models']}, Prediction CSV: {result['Prediction CSV']}")