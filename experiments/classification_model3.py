import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from sklearn.metrics import precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data[:, :-1]
    y = data[:, -1]
    print(np.unique(y, return_counts=True))
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        nan_mask = ~np.isnan(X).any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
    return X, y

def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    print(np.unique(y_test, return_counts=True))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer with 10% probability
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm(x)  # Apply batch normalization
        x = torch.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    test_precisions = []
    best_precision = 0
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Record training loss
        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate on test set
        if epoch % 10 == 0:
            precision, _, _ = evaluate_model(model, test_loader)
            test_precisions.append(precision)

            # Save the model with the highest precision
            if precision > best_precision:
                best_precision = precision
                best_model_state = model.state_dict()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Test Precision: {precision:.4f}')

    return train_losses, test_precisions, best_model_state

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate precision for the positive class
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    
    return precision, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main(file_path):
    # Load the data
    X, y = load_data(file_path)
    print(f"Original dataset shape: {X.shape, y.shape}")
    print(f"Sample data: {X[0]}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Balance the training data
    smote = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training dataset shape: {X_train_balanced.shape, y_train_balanced.shape}")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoader instances
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define the model, loss function, and optimizer
    input_dim = X_train_tensor.shape[1]
    hidden_dim = 64
    output_dim = len(np.unique(y))  # Number of classes
    model = MLPModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model and track metrics
    train_losses, test_precisions, best_model_state = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=1000)
    
    # Save the best model and scaler
    torch.save(best_model_state, './model_param/best_model.pth')
    dump(scaler, './model_param/scaler.joblib')
    
    # Plot the precision over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(test_precisions) + 1), test_precisions, marker='o', label='Test Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Test Precision Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Evaluate and plot the confusion matrix with the best model
    model.load_state_dict(best_model_state)
    _, y_pred, y_true = evaluate_model(model, test_loader)
    classes = [str(i) for i in range(output_dim)]  # Update this to your class names if necessary
    plot_confusion_matrix(y_true, y_pred, classes)

if __name__ == "__main__":
    file_path = './training_data/train_samples_24_6_5m.npy'
    main(file_path)
