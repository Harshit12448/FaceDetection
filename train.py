import os
import torch
from sklearn.metrics import accuracy_score


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import optuna
import torch.optim as optim

# Define dataset path
directory = r"C:\Users\singh\Downloads\archive\images\validation"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = datasets.ImageFolder(root=directory, transform=transform)

# Split into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader setup
batch_size = 32
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjusted for image size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)  # Assuming 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  

        x = torch.flatten(x, start_dim=1)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.softmax(self.fc3(x), dim=1)  

        return x

# Instantiate model
model = CNN()

# Loss function
criterion = nn.CrossEntropyLoss()


# Training function
def train_and_evaluate(model, optimizer):
    model.train()
    total_loss = 0

    for epoch in range(1):  # Single epoch for tuning
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(trainloader)



def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.5, 0.99)  
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Train and evaluate the model
    loss = train_and_evaluate(model, optimizer)

    return -loss  # Minimize loss (Optuna maximizes by default)

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("Finally initiated")
learning_rate = 0.0025216276995794324
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loss function
criterion = nn.CrossEntropyLoss()


def train_model(model, trainloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

# Testing function
def test_model(model, testloader):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Train the model
train_model(model, trainloader, optimizer, criterion, epochs=5)

# Test the model
# test_accuracy = test_model(model, testloader)
# Save the model's state_dict (weights and biases)
test_accuracy = test_model(model, testloader)
torch.save(model.state_dict(), 'cnn_model.pth')
