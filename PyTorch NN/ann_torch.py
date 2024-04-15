import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WineDataset(Dataset):
    def __init__(self, normalization=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.normalization = normalization
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0]).long() - 1
        
        if self.normalization:
            self.x = self.normalization(self.x)
        
        self.normalization = normalization
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample
    
    def __len__(self):
        return self.n_samples  

class Normalization:
    def __call__(self, x):
        mean = torch.mean(x, dim=0)
        std_dev = torch.std(x, dim=0)
        z = (x - mean) / std_dev
        return z
    

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.output(out)
        
        return out

dataset = WineDataset(normalization=Normalization())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.x, dataset.y, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

# Hyperparameters
epochs = 100
learning_rate = 0.001
input_size = dataset.x.shape[1]
num_classes = 3

model = NeuralNet(input_size, num_classes).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(epochs):
    # Training
    model.train()
    corrects, total_samples = 0, 0 
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = loss_func(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)  # Get predicted class indices
        corrects += (predicted == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Count total samples
        
    train_accuracy = (corrects / total_samples) * 100
    
    # Testing
    model.eval()
    corrects, total_samples = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
    test_accuracy = (corrects / total_samples) * 100
    
    if epoch % math.ceil(epochs/10) == 0:
        print(f'Epoch [{epoch}/{epochs}], Train Accuracy: {train_accuracy:.4f}%, Test Accuracy: {test_accuracy:.4f}%, Loss: {loss.item():.4f}')

print('Last Result:')
print("="*30)
print(f'| Train Accuracy: {train_accuracy:.4f}% | Test Accuracy: {test_accuracy:.4f}% | Loss: {loss.item():.4f} |')
print("="*30)