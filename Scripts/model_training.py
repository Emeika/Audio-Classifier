import pandas as pd
import numpy as np
from tensorflow import keras
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import load

# Load the test dataset
df_test = pd.read_csv('mfcc_features_test.csv')

# Assuming your test CSV has columns 'ID', 'FilePath', 'MFCC_1', 'MFCC_2', ..., 'MFCC_13'
X_test = df_test.iloc[:, 2:].values  # Extract features from the test dataset

# Reshape the input features for CNN and LSTM
X_test_cnn_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load the saved CNN model
loaded_cnn_model = keras.models.load_model('my_cnn_model.h5')

# Use the loaded CNN model for predictions on the test set
predictions_cnn = loaded_cnn_model.predict(X_test_cnn_lstm)

# Load the saved LSTM model
loaded_lstm_model = keras.models.load_model('my_lstm_model.h5')

# Use the loaded LSTM model for predictions on the test set
predictions_lstm = loaded_lstm_model.predict(X_test_cnn_lstm)

# Assuming you have already loaded your training dataset 'mfcc_features_train.csv'
df_train = pd.read_csv('mfcc_features_train.csv')

# Extract features and labels
X_train = df_train.iloc[:, 2:15].values  # Assuming columns 'MFCC_1' to 'MFCC_13' are your features
y_train = df_train['ClassID'].values

# Encode the class labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Reshape the input features for Random Forest and SVM
X_train_rf_svm = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test_rf_svm = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Load the saved Random Forest model
loaded_rf_model = load('my_random_forest_model.joblib')
predictions_rf = loaded_rf_model.predict_proba(X_test_rf_svm)

# Load the tuned XGBoost model
loaded_xgb_model_tuned = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)))
loaded_xgb_model_tuned.load_model('my_xgboost_model.json')
predictions_xgb_tuned = loaded_xgb_model_tuned.predict_proba(X_test_rf_svm)

# Combine predictions from all models using argmax
ensemble_predictions = np.argmax(predictions_cnn + predictions_lstm + predictions_rf + predictions_xgb_tuned, axis=1)

# Create a new dataframe with 'ID' and 'PredictedLabel' columns for the ensemble
df_result_ensemble = pd.DataFrame({'id': df_test['ID'], 'label': ensemble_predictions})

# Save the results of the ensemble to a new CSV file
df_result_ensemble.to_csv('predicted_results_ensemble.csv', index=False)
