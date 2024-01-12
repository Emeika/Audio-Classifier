import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras import layers  # Make sure to include this import
from keras.layers import LSTM
from joblib import dump

def build_and_train_model(X_train, y_train, model_type, epochs=250, batch_size=32):
    model = keras.Sequential()
    
    if model_type == 'cnn':
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
    elif model_type == 'lstm':
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
    else:
        raise ValueError("Invalid model type. Supported types are 'cnn' and 'lstm'.")
    
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model

# Load the dataset
df = pd.read_csv('mfcc_features_train.csv')

# Extract features and labels
X = df.iloc[:, 2:15].values 
y = df['ClassID'].values

# Encode the class labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input features for CNN and LSTM
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build, train, and save the CNN model
cnn_model = build_and_train_model(X_train_cnn, y_train, 'cnn', epochs=250, batch_size=32)
cnn_model.save('my_cnn_model.h5')

# Build, train, and save the LSTM model
lstm_model = build_and_train_model(X_train_cnn, y_train, 'lstm', epochs=250, batch_size=32)
lstm_model.save('my_lstm_model.h5')

# Build and train the XGBoost model
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)))
xgb_model.fit(X_train, y_train)

# Save the trained XGBoost model
xgb_model.save_model('my_xgboost_model.json')

# Build, train, and save a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained Random Forest model
dump(rf_model, 'my_random_forest_model.joblib')
