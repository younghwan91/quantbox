import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

# Load the data
load_data = np.load('./data/training_data/train_data.npy', allow_pickle=True)

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

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define a deeper DNN model
class DeepDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DeepDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

hidden_size = 64  # Adjust the hidden layer size for efficiency
input_size = X_train.shape[1]
num_classes = 2
num_epochs = 500
learning_rate = 0.001
batch_size = 32  # Adjust batch size as needed

model = DeepDNN(input_size, hidden_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Create DataLoader for batch training
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model with early stopping
patience = 1000
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test).item()
        
    # Check for early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    y_pred_proba = torch.softmax(outputs, dim=1)[:, 1].numpy()
    y_true = y_test.numpy()

# Function to evaluate different thresholds and calculate profit
def evaluate_thresholds_for_profit(y_true, y_pred_proba, thresholds, profit_per_tp, loss_per_fp):
    profits = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = tp * profit_per_tp - fp * loss_per_fp
        profits.append((threshold, profit))
    return profits

# Assume 1:1 win/loss ratio (profit per TP = loss per FP)
profit_per_tp = 100
loss_per_fp = 100

# Evaluate thresholds for profit
thresholds = np.arange(0.1, 1.0, 0.1)
profits = evaluate_thresholds_for_profit(y_true, y_pred_proba, thresholds, profit_per_tp, loss_per_fp)

# Find the threshold that maximizes profit
best_threshold, max_profit = max(profits, key=lambda x: x[1])
print(f"Best Threshold: {best_threshold}, Max Profit: {max_profit}")

# Evaluate the final model with the best threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_proba)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

print(f'Final Threshold: {best_threshold}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
