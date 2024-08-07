import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import dump
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data[:, :-1]
    y = data[:, -1]
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        nan_mask = ~np.isnan(X).any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
    return X, y

def balance_classes(X, y):
    # Balance the classes using undersampling
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X, y)
    return X_balanced, y_balanced

def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def train_with_grid_search(X_train, y_train):
    base_estimator = RandomForestClassifier(random_state=42)
    param_grid = {
        'estimator__n_estimators': [50, 100],
        'estimator__max_depth': [3, 5],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2],
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    }
    ada = AdaBoostClassifier(estimator=base_estimator, random_state=42, algorithm='SAMME')
    grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=3, scoring='precision', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    best_model = grid_search.best_estimator_
    return best_model

def train_with_default_params(X_train, y_train):
    ada = AdaBoostClassifier(estimator=RandomForestClassifier(random_state=42, max_depth= 15, min_samples_leaf=1, min_samples_split=2, n_estimators=25), 
                             random_state=42, n_estimators=50, learning_rate=0.6, algorithm='SAMME')
    ada.fit(X_train, y_train)
    # feature importance
    feature_importance = ada.feature_importances_
    print(f"Feature importance: {feature_importance}")
    return ada

def evaluate_model(model, X_test, y_test, confidence_threshold):
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    confidence = np.max(y_proba, axis=1)

    # Apply confidence threshold
    confident_mask = confidence >= confidence_threshold
    y_pred_confident = y_pred[confident_mask]
    y_test_confident = y_test[confident_mask]

    if len(y_pred_confident) > 0:
        accuracy = accuracy_score(y_test_confident, y_pred_confident)
        precision = precision_score(y_test_confident, y_pred_confident, average='weighted', zero_division=0)
        recall = recall_score(y_test_confident, y_pred_confident, average='weighted', zero_division=0)
        f1 = f1_score(y_test_confident, y_pred_confident, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test_confident, y_pred_confident)
        class_report = classification_report(y_test_confident, y_pred_confident)
    else:
        accuracy = precision = recall = f1 = 0
        conf_matrix = np.array([])
        class_report = "No predictions met the confidence threshold."

    return accuracy, precision, recall, f1, conf_matrix, class_report

def main(file_path, use_grid_search=True):
    # Load the data
    X, y = load_data(file_path)
    print(f"Original dataset shape: {X.shape, y.shape}")
    print(f"Sample data: {X[0]}")
    
    # count classes in y
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    # Split the data into training and test sets before SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE only on the training data
    smote = ADASYN(random_state=42)
    # smote = AllKNN()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training dataset shape: {X_train_balanced.shape, y_train_balanced.shape}")

    # Scale the training and test data
    scaler = StandardScaler().fit(X_train_balanced)
    X_train_scaled = scaler.transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Train the model with the balanced training data
    if use_grid_search:
        model = train_with_grid_search(X_train_scaled, y_train_balanced)
    else:
        model = train_with_default_params(X_train_scaled, y_train_balanced)
    
    # Evaluate the model using the original test data
    accuracy, precision, recall, f1, conf_matrix, class_report = evaluate_model(model, X_test_scaled, y_test, confidence_threshold=0.5)

    # Save the model and scaler
    dump(model, './model_param/trained_model.joblib')
    dump(scaler, './model_param/scaler.joblib')
    
    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

if __name__ == "__main__":
    file_path = './training_data/train_samples_24_6_5m.npy'
    
    main(file_path, use_grid_search=0)  # Change to True to use grid search
