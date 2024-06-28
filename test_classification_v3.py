import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from joblib import dump

# Load the data
load_data = np.load('./data/training_data/train_data_2days_1hrs.npy', allow_pickle=True)

# Split the data into features (X) and labels (y)
X = load_data[:, :-1]
y = load_data[:, -1]

# Check for and remove rows containing NaN values
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    nan_mask = ~np.isnan(X).any(axis=1)
    X = X[nan_mask]
    y = y[nan_mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the base estimator
base_estimator = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'estimator__n_estimators': [50, 100],
    'estimator__max_depth': [3, 5],
    'estimator__min_samples_split': [2, 5],
    'estimator__min_samples_leaf': [1, 2, 4],
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 1.0]
}

# Initialize the AdaBoost classifier with RandomForest as the base estimator
ada = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=5, scoring='precision', n_jobs=-1, verbose=2)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the final model with the best parameters
best_model = grid_search.best_estimator_

# Save the best model, scaler, and parameters
dump(best_model, 'trained_model.joblib')
dump(scaler, 'scaler.joblib')

# Predict probabilities
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_true = y_test

# Evaluate the final model with the best threshold
y_pred = (y_pred_proba >= 0.5).astype(int)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)