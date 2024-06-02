import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train'
test_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Test'
train_dataset = ImageFolder(root=train_image_root, transform=transform)
test_loader = ImageFolder(root=test_image_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True)
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_to_lin, lin_to_lin, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_size2, out_channels=hidden_size3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=hidden_size3, out_channels=hidden_to_lin, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_to_lin, lin_to_lin)
        self.fc2 = nn.Linear(lin_to_lin, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 36)))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


input_size = 1
hidden_size1 = 9
hidden_size2 = 18
hidden_size3 = 27
hidden_to_lin = 36
lin_to_lin = 100
output_size = 2
batch_size = 3
learning_rate = 0.01
num_epochs = 1000


model = CNN(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_to_lin, lin_to_lin, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(500):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 10 == 9:
        print('[%d] loss: %.10f' % (epoch + 1, running_loss / 100))
        running_loss = 0.0

model.eval()
torch.save(model.state_dict(), 'model.pth')
correct = 0
count = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.unsqueeze(1)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        print("예측률 :", output[0][0], ",", output[0][1])
        correct += pred.eq(torch.LongTensor([target])).sum().item()
        count+=1
print(correct/count)