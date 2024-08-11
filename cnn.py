import os
from sklearn.model_selection import train_test_split
import cv2
import torch.nn as nn
from torchsummary import summary
import torch
from torch.utils.data import DataLoader, Dataset, random_split

file_dir = 'GTSRB_subset_new'

class_names = ["class1", "class2"]
dataset = []

for label, cls in enumerate(class_names):
    path = os.path.join(file_dir, cls)

    class_images = [
        (cv2.imread(os.path.join(path, img)) / 255).reshape(3, 64, 64)
        for img in os.listdir(path)
    ]
    dataset.extend([(img_array, label) for img_array in class_images])

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(90, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


modelCNN = CNN()


print(modelCNN)

summary(modelCNN,input_size=(3,64,64))

epoch_cnn = 20
lose_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelCNN.parameters(), lr=0.01)

def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

train(modelCNN, train_loader, optimizer, lose_function, epoch_cnn)

def evaluate(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    print(f"Accuracy on test set: {accuracy:.2f}%")

print()
evaluate(modelCNN, test_loader, lose_function)