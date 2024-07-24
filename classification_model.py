import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import dump

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data[:, :-1]
    y = data[:, -1]
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        nan_mask = ~np.isnan(X).any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
    return X, y

def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    ada = AdaBoostClassifier(estimator=base_estimator, random_state=42)
    grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    best_model = grid_search.best_estimator_
    return best_model

def train_with_default_params(X_train, y_train):
    ada = AdaBoostClassifier(estimator=RandomForestClassifier(random_state=42), random_state=42)
    ada.fit(X_train, y_train)
    return ada

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix, class_report

def main(file_path, use_grid_search=True):
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    if use_grid_search:
        model = train_with_grid_search(X_train, y_train)
    else:
        model = train_with_default_params(X_train, y_train)
    accuracy, precision, recall, f1, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    dump(model, './model_param/trained_model.joblib')
    dump(scaler, './model_param/scaler.joblib')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

if __name__ == "__main__":
    file_path = './training_data/train_data_48_4.npy'
    main(file_path, use_grid_search=False)  # Change to True to use grid search

