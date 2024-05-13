import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, skiprows=[0])
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values

        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


input_size = 34
hidden_size = 64
output_size = 5
batch_size = 16
learning_rate = 0.001
num_epochs = 1000


dataset = CustomDataset('captured_images/pos_data.csv')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.33, random_state=42, shuffle=True)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = MLP(input_size, hidden_size, output_size)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Training Finished')


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    accuracy = correct / total
    return accuracy, predictions, true_labels

eval_index = 90

eval_data = dataset[eval_index]
eval_loader = DataLoader(dataset=[eval_data], batch_size=1)

eval_accuracy, eval_predictions, eval_true_labels = evaluate(model, eval_loader)
print(f'Accuracy of the network on the evaluation image: {100 * eval_accuracy:.2f}%')

print("True Label:", eval_true_labels[0])
print("Prediction:", eval_predictions[0])
torch.save(model.state_dict(), 'model.pth')