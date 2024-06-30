import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import argparse
import numpy as np
from torch import Tensor
from typing import Optional


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((300, 500)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train'
test_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Test'


train_dataset = ImageFolder(root=train_image_root, transform=transform)
test_dataset= ImageFolder(root=test_image_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True)


class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj) -> None:
        super(Inception, self).__init__()
        self.branch1 = ConvBlock(in_channels, n1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n3x3_reduce, n3x3, kernel_size=3, stride=1, padding=1))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n5x5_reduce, n5x5, kernel_size=5, stride=1, padding=2))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0))

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000) -> None:
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def transform_input(self, x: Tensor) -> Tensor:
        x_R = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_G = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_B = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat([x_R, x_G, x_B], 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform_input(x)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool3(x)
        x = self.a4(x)
        aux1: Optional[Tensor] = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        aux2: Optional[Tensor] = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.e4(x)
        x = self.maxpool4(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.dropout(x)

        if self.aux_logits and self.training:
            return aux1, aux2
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet(aux_logits=True, num_classes=3)
    print(model(x)[1].shape)


if __name__ == "__main__":
    # set hyperparameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, default=100)
    parser.add_argument('--learning_rate', action='store', type=float, default='0.0002')
    parser.add_argument('--n_epochs', action='store', type=int, default=100)
    parser.add_argument('--plot', action='store', type=bool, default=True)
    args = parser.parse_args()

    np.random.seed(1)
    seed = torch.manual_seed(1)

    # model, loss, optimizer
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(aux_logits=False, num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            count += labels.size(0)
            correct += preds.eq(labels).sum().item()  # torch.sum(preds == labels)

            if batch_idx % 1 == 0:
                print(
                    f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_loader)}\tTrain accuracy: {round((correct / count), 4)} \tTrain Loss: {round((train_loss / count) * 100, 4)}")

        # valid
        model.eval()
        correct, count = 0, 0
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                _, preds = torch.max(output, 1)
                count += labels.size(0)
                correct += preds.eq(labels).sum().item()  # torch.sum(preds == labels)
                if batch_idx % 100 == 0:
                    print(
                        f"[*] Step: {batch_idx}/{len(test_loader)}\tValid accuracy: {round((correct / count), 4)} \tValid Loss: {round((valid_loss / count) * 100, 4)}")
torch.save(model.state_dict(), 'model_google.pth')
