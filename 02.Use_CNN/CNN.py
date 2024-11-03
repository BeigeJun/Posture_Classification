import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/02.Use_CNN/Train'
test_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/02.Use_CNN/Test'
train_dataset = ImageFolder(root=train_image_root, transform=transform)
test_dataset = ImageFolder(root=test_image_root, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

class CNN(nn.Module):
    def __init__(self, input_channels, in_to_hidden, lin_to_lin, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=in_to_hidden, kernel_size=3, stride=1,
                               padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)

        # Feature map 크기를 계산하기 위해 임의의 입력 크기를 사용합니다.
        self.fc_input_size = in_to_hidden * 25 * 25  # 입력 크기 200x200 이미지를 4x4로 pooling 했을 때

        self.fc1 = nn.Linear(self.fc_input_size, lin_to_lin)
        self.fc2 = nn.Linear(lin_to_lin, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_channels = 1
in_to_hidden = 16
lin_to_lin = 512
output_size = 2
batch_size = 12
learning_rate = 0.01
num_epochs = 100

model = CNN(input_channels, in_to_hidden, lin_to_lin, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 9:
        print('[%d] loss: %.100f' % (epoch + 1, running_loss / 100))
        running_loss = 0.0

# 모델 저장
model.eval()
torch.save(model.state_dict(), 'model.pth')


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(labels.cpu(), outputs.cpu())
print('Test Accuracy: {:.2f}%'.format(100 * correct / total))