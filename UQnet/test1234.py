import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformation for the dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load dataset (MNIST for simplicity)
x_train = torch.linspace(0, 1, 100)
# add noise to sin function
y_train = torch.sin(x_train) + torch.randn(x_train.size())



# Define the first model
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the second model
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Training function
def train(model, optimizer, epoch):
    data, target = x_train, y_train
    def closure():
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        return loss
    
    for epoch in range(1, 10):
        optimizer.step(closure)
        loss = closure()
        print(
            f"Training: Epoch [{epoch + 1}/{10}], "
            f"Training Loss: {loss.item():.4f}"
        )


# Main function to train two models sequentially
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First model
    model1 = Model1().to(device)
    # optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
    optimizer1 = torch.optim.LBFGS(model1.parameters(), lr=0.1)
    for epoch in range(1, 11):
        train(model1, optimizer1, epoch)

    # Second model
    model2 = Model2().to(device)
    # optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    optimizer2 = torch.optim.LBFGS(model2.parameters(), lr=0.1)
    for epoch in range(1, 11):
        train(model2, optimizer2, epoch)


if __name__ == "__main__":
    main()
