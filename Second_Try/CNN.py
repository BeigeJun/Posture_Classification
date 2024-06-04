import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train'
test_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Test'

train_dataset = ImageFolder(root=train_image_root, transform=transform)
test_dataset = ImageFolder(root=test_image_root, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_size1, hidden_to_lin, lin_to_lin, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_size1, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=hidden_size1, out_channels=hidden_to_lin, kernel_size=7, stride=1, padding=3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._to_linear = None
        self.convs(torch.randn(1, input_channels, 200, 200))

        self.fc1 = nn.Linear(self._to_linear, lin_to_lin)
        self.fc2 = nn.Linear(lin_to_lin, output_size)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).size(1)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

input_size = 1
hidden_size1 = 16
hidden_size2 = 32
hidden_size3 = 64
hidden_to_lin = 256
lin_to_lin = 512
output_size = 2
batch_size = 12
learning_rate = 0.01
num_epochs = 100


model = CNN(input_size, hidden_size1, hidden_to_lin, lin_to_lin, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
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

with torch.no_grad():
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    First_conv_outputs = F.relu(model.conv1(images))
    First_pool_outputs = F.max_pool2d(First_conv_outputs, kernel_size=2, stride=2)
    Second_conv_outputs = F.relu(model.conv2(First_pool_outputs))
    Second_pool_outputs = F.max_pool2d(Second_conv_outputs, kernel_size=2, stride=2)
    conv1_weights = model.conv1.weight.data
    print(conv1_weights)
    print(conv1_weights.shape)
    conv2_weights = model.conv2.weight.data
    print(conv2_weights)
    print(conv2_weights.shape)

    import matplotlib.pyplot as plt
    plt.imshow(images[0][0].detach().numpy(), cmap='gray')
    plt.title("Original Image")
    plt.show()
    #앞이 배치 순서, 뒤가 배치에서의 순서
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(First_conv_outputs[0][i].squeeze().detach().numpy(), cmap='gray')
        plt.title("First Convolution Output")
    plt.show()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(First_pool_outputs[0][i].squeeze().detach().numpy(), cmap='gray')
        plt.title("First Pooling Output")
    plt.show()
    for i in range(18):
        plt.subplot(3, 6, i + 1)
        plt.imshow(Second_conv_outputs[0][i].squeeze().detach().numpy(), cmap='gray')
        plt.title("Second Convolution Output")
    plt.show()
    for i in range(18):
        plt.subplot(3, 6, i + 1)
        plt.imshow(Second_pool_outputs[0][i].squeeze().detach().numpy(), cmap='gray')
        plt.title("Second Pooling Output")
    plt.show()
