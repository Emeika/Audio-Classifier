import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

# Load the augmented training data
train_data = pd.read_csv('../data/stratified_training_set.csv')
val_data = pd.read_csv('../data/stratified_validation_set.csv')

# Split the data into features and labels
X_train = train_data.iloc[:, 2:-1]  # Assuming the features start from the 3rd column
y_train = train_data['ClassID']

X_val = val_data.iloc[:, 2:-1]  # Assuming the features start from the 3rd column
y_val = val_data['ClassID']

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Count the samples in each class
class_counts = np.bincount(y_train)

# Determine the class with the maximum count
majority_class = np.argmax(class_counts)

# Determine the ratio of samples to replicate
replication_ratio = class_counts[majority_class] / class_counts

# Replicate samples from minority classes to balance the class distribution
X_replicated = [X_train_scaled[y_train == cls].repeat(replication_ratio[cls], axis=0) for cls in range(10)]
X_train_balanced = np.vstack(X_replicated)
y_train_balanced = np.concatenate([np.full(len(X_replicated[cls]), cls) for cls in range(10)])

# Shuffle the balanced training set
shuffle_indices = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced[shuffle_indices]
y_train_balanced = y_train_balanced[shuffle_indices]

# Create and train the MLP model with additional strategies
mlp = MLPClassifier(
    hidden_layer_sizes=(200, 100),    # Two hidden layers with 200 and 100 neurons
    activation='relu',                # Rectified Linear Unit (ReLU) activation function
    solver='adam',                    # Adam optimization algorithm
    alpha=0.0001,                     # L2 regularization term
    batch_size='auto',                # Automatic batch size determination
    learning_rate='adaptive',         # Adaptive learning rate
    max_iter=200,                      # Maximum number of iterations
    early_stopping=True,              # Enable early stopping
    validation_fraction=0.1,          # Fraction of training data to use as a validation set
    n_iter_no_change=10,              # Number of iterations with no improvement to wait for early stopping
    random_state=42
)

mlp.fit(X_train_balanced, y_train_balanced)

# Make predictions on the validation set
y_pred = mlp.predict(X_val_scaled)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# Print classification report for more detailed metrics
print(classification_report(y_val, y_pred))

# Save the trained model and the scaler
joblib.dump(mlp, '../results/models/mlp_model.pkl')
joblib.dump(scaler, '../results/scalers/mlp_scaler.pkl')
