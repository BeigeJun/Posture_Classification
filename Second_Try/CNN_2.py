import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((300, 500)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train'
test_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Test'


train_dataset = ImageFolder(root=train_image_root, transform=transform)
test_dataset= ImageFolder(root=test_image_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(75 * 125 * 18, 100)
        self.fc2 = nn.Linear(100, 3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 75 * 125 * 18)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

cnn = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss() #LogSoftmax를 포함하고 있다.
optimizer = optim.SGD(cnn.parameters(), lr=0.001)

cnn.train()
for epoch in range(10000):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = cnn(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 10 == 9:
        print('[%d] loss: %.10f' % (epoch + 1, running_loss / 100))
        running_loss = 0.0


cnn.eval()
torch.save(cnn.state_dict(), 'model.pth')

correct = 0
count = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = cnn(data)
        pred = output.argmax(dim=1, keepdim=True)

        print("예측률 :", output[0][0], ",", output[0][1])
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += 1
print(correct / count)
