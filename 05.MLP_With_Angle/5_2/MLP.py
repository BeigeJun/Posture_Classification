import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


csv_file_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Angle_data.csv'
data = pd.read_csv(csv_file_path)

X_12 = data.iloc[:, :-2].values
X_13 = data.iloc[:, -2].values
y = data['label'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_12_train, X_12_test, X_13_train, X_13_test, y_train, y_test = train_test_split(X_12, X_13, y, test_size=0.2, random_state=42)

X_12_train_tensor = torch.tensor(X_12_train, dtype=torch.float32)
X_13_train_tensor = torch.tensor(X_13_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_12_test_tensor = torch.tensor(X_12_test, dtype=torch.float32)
X_13_test_tensor = torch.tensor(X_13_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_12_train_tensor, X_13_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_12_test_tensor, X_13_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(33, num_classes)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.2)

    def forward(self, x_12, x_13):
        out = self.dropout1(x_12)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = torch.cat((out, x_13), dim=1)  # 64차원 + 1차원 (13번째 각도)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        return out



input_size = X_12_train.shape[1]
num_classes = len(label_encoder.classes_)
model = MLP(input_size=input_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 1000000


for epoch in range(num_epochs):
    for i, (inputs_12, input_13, labels) in enumerate(train_loader):
        outputs = model(inputs_12, input_13)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.10f}')


model_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/First_MLP.pth'
torch.save(model.state_dict(), model_path)


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs_12, input_13, labels in test_loader:
        outputs = model(inputs_12, input_13)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
