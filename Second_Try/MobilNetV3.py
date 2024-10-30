import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(device)
    print(start_time_str)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train'
    val_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Validation'
    test_image_root = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Test'

    train_dataset = ImageFolder(root=train_image_root, transform=transform)
    val_dataset = ImageFolder(root=val_image_root, transform=transform)
    test_dataset = ImageFolder(root=test_image_root, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=4,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,shuffle=False)

    def _make_divisible(v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return int(new_v)

    class h_swish(nn.Module):
        def __init__(self):
            super(h_swish, self).__init__()
            self.relu6 = nn.ReLU6()
        def forward(self, x):
            return x * (self.relu6(x + 3) / 6)

    class inverted_residual_block(nn.Module):
        def __init__(self, i, t, o, k, s, re=False, se=False):
            super(inverted_residual_block, self).__init__()
            expansion = int(i * t)
            if re:
                nonlinear = nn.ReLU6()
            else:
                nonlinear = h_swish()
            self.se = se
            self.conv = nn.Sequential(
                nn.Conv2d(i, expansion, 1, 1),
                nn.BatchNorm2d(expansion),
                nonlinear
            )
            self.dconv = nn.Sequential(
                nn.Conv2d(expansion, expansion, k, s, k // 2, groups=expansion),
                nn.BatchNorm2d(expansion),
                nonlinear
            )
            self.semodule = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expansion, _make_divisible(expansion // 4), 1, 1),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(expansion // 4), expansion, 1, 1),
                h_swish()
            )
            self.linearconv = nn.Sequential(
                nn.Conv2d(expansion, o, 1, 1),
                nn.BatchNorm2d(o)
            )
            self.shortcut = (i == o and s == 1)

        def forward(self, x):
            out = self.conv(x)
            out = self.dconv(out)
            if self.se:
                out *= self.semodule(out)
            out = self.linearconv(out)
            if self.shortcut:
                out += x
            return out

    class mobilenetv3(nn.Module):
        def __init__(self, ver=1, w=1.0):
            super(mobilenetv3, self).__init__()
            large = [
                [1, 16, 3, 1, False, False],
                [4, 24, 3, 2, False, False],
                [3, 24, 3, 1, False, False],
                [3, 40, 5, 2, False, True],
                [3, 40, 5, 1, False, True],
                [3, 40, 5, 1, False, True],
                [6, 80, 3, 2, True, False],
                [2.5, 80, 3, 1, True, False],
                [2.4, 80, 3, 1, True, False],
                [2.4, 80, 3, 1, True, False],
                [6, 112, 3, 1, True, True],
                [6, 112, 3, 1, True, True],
                [6, 160, 5, 2, True, True],
                [6, 160, 5, 1, True, True],
                [6, 160, 5, 1, True, True]
            ]

            small = [
                [1, 16, 3, 2, False, True],
                [4, 24, 3, 2, False, False],
                [11.0 / 3.0, 24, 3, 1, False, False],
                [4, 40, 5, 2, True, True],
                [6, 40, 5, 1, True, True],
                [6, 40, 5, 1, True, True],
                [3, 48, 5, 1, True, True],
                [3, 48, 5, 1, True, True],
                [6, 96, 5, 2, True, True],
                [6, 96, 5, 1, True, True],
                [6, 96, 5, 1, True, True],
            ]

            in_channels = _make_divisible(16 * w)

            self.conv1 = nn.Sequential(
                nn.Conv2d(3, in_channels, 3, 2, 1),
                nn.BatchNorm2d(int(16 * w)),
                nn.ReLU6()
            )
            if ver == 0:
                stack = large
                last = 1280
            else:
                stack = small
                last = 1024
            layers = []

            for i in range(len(stack)):
                out_channels = _make_divisible(stack[i][1] * w)
                layers.append(
                    inverted_residual_block(in_channels, stack[i][0], out_channels, stack[i][2], stack[i][3], stack[i][4],
                                            stack[i][5]))
                in_channels = out_channels
            self.stack = nn.Sequential(*layers)
            self.last = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 6, 1, 1),
                nn.BatchNorm2d(out_channels * 6),
                h_swish(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels * 6, last, 1, 1),
                h_swish(),
                nn.Conv2d(last, 6, 1, 1)
            )

        def forward(self, x):
            out = self.conv1(x)
            out = self.stack(out)
            out = self.last(out)
            out = out.view(out.size(0), -1)
            return out

    cnn = mobilenetv3().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    num_epochs = 5000
    patience = 1000
    patience_count = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    Top_Accuracy_Train = 0
    Top_Accuracy_Validation = 0
    Top_Accuracy_Train_Epoch = 0
    Top_Accuracy_Validation_Epoch = 0

    Bottom_Loss_Train = float('inf')
    Bottom_Loss_Validation = float('inf')
    Bottom_Loss_Train_Epoch = 0
    Bottom_Loss_Validation_Epoch = 0

    save_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/MobileNet_Save/Small/'
    os.makedirs(save_path, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        cnn.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        if patience_count >= patience:
            plt.savefig(save_path + 'training_validation_graphs.png')
            plt.close()
            break

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        patience_count += 1
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        if Top_Accuracy_Train < train_accuracy:
            Top_Accuracy_Train = train_accuracy
            model_path_make = save_path + 'Best_Accuracy_Train_MLP.pth'
            torch.save(cnn.state_dict(), model_path_make)
            Top_Accuracy_Train_Epoch = epoch

        if Bottom_Loss_Train > running_loss:
            Bottom_Loss_Train = running_loss
            model_path_make = save_path + 'Bottom_Loss_Train_MLP.pth'
            torch.save(cnn.state_dict(), model_path_make)
            Bottom_Loss_Train_Epoch = epoch

        if (epoch + 1) % 10 == 0:
            train_losses.append(running_loss / len(train_loader))
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            cnn.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = cnn(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_losses.append(val_loss / len(val_loader))
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

            axs[0].clear()
            axs[1].clear()

            axs[0].plot(range(10, epoch + 2, 10), train_accuracies, label='Train Accuracy', color='red', linewidth=0.5)
            axs[0].plot(range(10, epoch + 2, 10), val_accuracies, label='Validation Accuracy', color='blue', linewidth=0.5)
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title('Training and Validation Accuracy')
            axs[0].legend()

            axs[1].plot(range(10, epoch + 2, 10), train_losses, label='Train Loss', color='red', linewidth=0.5)
            axs[1].plot(range(10, epoch + 2, 10), val_losses, label='Validation Loss', color='blue', linewidth=0.5)
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Loss')
            axs[1].set_title('Training and Validation Loss')
            axs[1].legend()

            for tick in axs[0].get_xticks():
                axs[0].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

            for tick in axs[0].get_yticks():
                axs[0].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

            for tick in axs[1].get_xticks():
                axs[1].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

            for tick in axs[1].get_yticks():
                axs[1].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

            plt.draw()
            plt.pause(0.1)

            if Bottom_Loss_Validation > val_loss:
                Bottom_Loss_Validation = val_loss
                model_path_make = save_path + 'Bottom_Loss_Validation_MLP.pth'
                torch.save(cnn.state_dict(), model_path_make)
                Bottom_Loss_Validation_Epoch = epoch
                patience_count = 0

            if Top_Accuracy_Validation < val_accuracy:
                Top_Accuracy_Validation = val_accuracy
                model_path_make = save_path + 'Best_Accuracy_Validation_MLP.pth'
                torch.save(cnn.state_dict(), model_path_make)
                Top_Accuracy_Validation_Epoch = epoch

            if (epoch + 1) % 50 == 0:
                plt.savefig(save_path + 'training_validation_graphs.png')

                elapsed_training_time = time.time()
                elapsed_training_time = elapsed_training_time - start_time

                with open(save_path + 'numbers.txt', "w") as file:
                    file.write(f"Top Accuracy Train Epoch : {Top_Accuracy_Train_Epoch} Accuracy : {Top_Accuracy_Train}\n"
                               f"Top Accuracy Validation Epoch : {Top_Accuracy_Validation_Epoch} Accuracy : {Top_Accuracy_Validation}\n"
                               f"Bottom Loss Train Epoch : {Bottom_Loss_Train_Epoch} Loss : {Bottom_Loss_Train}\n"
                               f"Bottom Loss Validation Epoch : {Bottom_Loss_Validation_Epoch} Loss : {Bottom_Loss_Validation}\n"
                               f"Elapsed Time: {elapsed_training_time // 60} min {elapsed_training_time % 60:.2f} sec\n"
                               f"Patience Count : {patience_count}/{patience}\n")

            end_time = time.time()
            elapsed_time = end_time - start_time

    with open(save_path + 'numbers.txt', "w") as file:
        file.write(f"Top Accuracy Train Epoch : {Top_Accuracy_Train_Epoch} Accuracy : {Top_Accuracy_Train}\n"
                   f"Top Accuracy Validation Epoch : {Top_Accuracy_Validation_Epoch} Accuracy : {Top_Accuracy_Validation}\n"
                   f"Bottom Loss Train Epoch : {Bottom_Loss_Train_Epoch} Loss : {Bottom_Loss_Train}\n"
                   f"Bottom Loss Validation Epoch : {Bottom_Loss_Validation_Epoch} Loss : {Bottom_Loss_Validation}\n"
                   f"Elapsed Time: {elapsed_time // 60} min {elapsed_time % 60:.2f} sec\n"
                   f"Patience Count : {patience_count}/{patience}\n")
    cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    with open(save_path + 'numbers.txt', "a") as file:
        file.write(f"Test Accuracy: {test_accuracy:.4f}\n")

if __name__ == '__main__':
    main()