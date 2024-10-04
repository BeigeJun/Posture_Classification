import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


data = pd.read_csv('labeled_data.csv')

state_mapping = {
    'FallDown': 0,
    'FallingDown': 1,
    'Sit_chair': 2,
    'Sit_floor': 3,
    'Sleep': 4,
    'Stand': 5,
    'Terrified': 6
}
data['Input1'] = data['Input1'].map(state_mapping)
data['Input2'] = data['Input2'].map(state_mapping)
data['Input3'] = data['Input3'].map(state_mapping)

label_mapping = {
    'Danger': [1, 0, 0],
    'Caution': [0, 1, 0],
    'Safe': [0, 0, 1]
}
data['Label'] = data['Label'].map(label_mapping)

data = data[data['Label'].apply(lambda x: isinstance(x, list) and len(x) == 3)]

inputs = torch.tensor(data[['Input1', 'Input2', 'Input3']].values, dtype=torch.float32).view(-1, 3, 1)
labels = torch.tensor(data['Label'].tolist(), dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)


class PostureRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PostureRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size)
        c_0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 64
output_size = 3
model = PostureRNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train_model(model, inputs, labels, epochs=100000):
    for epoch in range(epochs):
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')


train_model(model, X_train, y_train)
torch.save(model.state_dict(), 'posture_rnn_model.pth')

model.eval()
with torch.no_grad():
    test_output = model(X_test)
    predicted = F.softmax(test_output, dim=1)

predicted_classes = torch.argmax(predicted, dim=1)
true_classes = torch.argmax(y_test, dim=1)

accuracy = (predicted_classes == true_classes).float().mean() * 100
print(f'Accuracy: {accuracy.item():.2f}%')
