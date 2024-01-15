import pandas as pd
from sklearn.model_selection import train_test_split

# Load your labeled dataset from a CSV file
file_path = 'mfcc_features_train.csv'
data = pd.read_csv(file_path)

# Assuming the 'ClassID' column contains your class labels
X = data.drop('ClassID', axis=1)  # Features
y = data['ClassID']  # Target variable

# Perform stratified split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create new DataFrames for training and validation sets
train_set = pd.concat([X_train, y_train], axis=1)
val_set = pd.concat([X_val, y_val], axis=1)

# Save the stratified training and validation sets to CSV files
train_set.to_csv('stratified_training_set.csv', index=False)
val_set.to_csv('stratified_validation_set.csv', index=False)
