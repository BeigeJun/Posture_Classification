import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import openpyxl
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Basic_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/'
csv_file_path = Basic_path + '05.MLP_With_Angle/Data/Angle_data.csv'
Exel_file_path = Basic_path + '06.Find_The_Best/Parameter.xlsx'
D_Name = '/No_DropOut'
num_epochs = 500000
patience = 100000

data = pd.read_csv(csv_file_path)

X = data.drop('label', axis=1).values
y = data['label'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size, f1_num, f2_num, f3_num, f4_num, f5_num, f6_num, d1, d2, d3, d4, d5, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, f1_num)
        self.fc2 = nn.Linear(f1_num, f2_num)
        self.fc3 = nn.Linear(f2_num, f3_num)
        self.fc4 = nn.Linear(f3_num, f4_num)
        self.fc5 = nn.Linear(f4_num, f5_num)
        self.fc6 = nn.Linear(f5_num, f6_num)
        self.fc7 = nn.Linear(f6_num, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        return out

input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)

Exel_File = openpyxl.load_workbook(Exel_file_path)
Sheet_Read_Data = Exel_File['Parameters']
Sheet_Save_Data = Exel_File['Results']

for index, (input_num, h1, h2, h3, h4, h5, h6, output_num, d1, d2, d3, d4, d5, batch, optimizer, lr) in enumerate(Sheet_Read_Data.iter_rows(values_only=True)):
    if index == 0:
        continue

    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(start_time_str)

    model = MLP(input_size, int(h1), int(h2), int(h3), int(h4), int(h5), int(h6), float(d1), float(d2), float(d3),
                float(d4), float(d5), int(output_num)).to(device)

    model_path = Basic_path + 'ModelsSave' + D_Name + '/' + str(index) + '/'
    os.makedirs(model_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    opt = optimizer == 'Adam'
    if opt:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    patience_count = 0

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    Top_Accuracy_Train = 0
    Top_Accuracy_Validation = 0
    Top_Accuracy_Train_Epoch = 0
    Top_Accuracy_Validation_Epoch = 0

    Bottom_Loss_Train = 1000
    Bottom_Loss_Validation = 1000
    Bottom_Loss_Train_Epoch = 0
    Bottom_Loss_Validation_Epoch = 0

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        if patience_count >= patience:
            plt.savefig(model_path + 'training_validation_graphs.png')
            plt.close()
            break

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        patience_count += 1
        train_accuracy = correct_train / total_train

        if Top_Accuracy_Train < train_accuracy:
            Top_Accuracy_Train = train_accuracy
            model_path_make = model_path + 'Best_Accuracy_Train_MLP.pth'
            torch.save(model.state_dict(), model_path_make)
            Top_Accuracy_Train_Epoch = epoch

        if Bottom_Loss_Train > running_loss:
            Bottom_Loss_Train = running_loss
            model_path_make = model_path + 'Bottom_Loss_Train_MLP.pth'
            torch.save(model.state_dict(), model_path_make)
            Bottom_Loss_Train_Epoch = epoch

        if (epoch + 1) % 100 == 0:
            train_losses.append(running_loss / len(train_loader))
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
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

            axs[0].plot(range(100, epoch + 2, 100), train_accuracies, label='Train Accuracy', color='red', linewidth=0.5)
            axs[0].plot(range(100, epoch + 2, 100), val_accuracies, label='Validation Accuracy', color='blue', linewidth=0.5)
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title('Training and Validation Accuracy')
            axs[0].legend()

            axs[1].plot(range(100, epoch + 2, 100), train_losses, label='Train Loss', color='red', linewidth=0.5)
            axs[1].plot(range(100, epoch + 2, 100), val_losses, label='Validation Loss', color='blue', linewidth=0.5)
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
                model_path_make = model_path + 'Bottom_Loss_Validation_MLP.pth'
                torch.save(model.state_dict(), model_path_make)
                Bottom_Loss_Validation_Epoch = epoch
                patience_count = 0

            if Top_Accuracy_Validation < val_accuracy:
                Top_Accuracy_Validation = val_accuracy
                model_path_make = model_path + 'Best_Accuracy_Validation_MLP.pth'
                torch.save(model.state_dict(), model_path_make)
                Top_Accuracy_Validation_Epoch = epoch

        if (epoch + 1) % 1000 == 0:
            plt.savefig(model_path + 'training_validation_graphs.png')

            elapsed_training_time = time.time()
            elapsed_training_time = elapsed_training_time - start_time

            with open(model_path + 'numbers.txt', "w") as file:
                file.write(f"Top Accuracy Train Epoch : {Top_Accuracy_Train_Epoch} Accuracy : {Top_Accuracy_Train}\n"
                           f"Top Accuracy Validation Epoch : {Top_Accuracy_Validation_Epoch} Accuracy : {Top_Accuracy_Validation}\n"
                           f"Bottom Loss Train Epoch : {Bottom_Loss_Train_Epoch} Loss : {Bottom_Loss_Train}\n"
                           f"Bottom Loss Validation Epoch : {Bottom_Loss_Validation_Epoch} Loss : {Bottom_Loss_Validation}\n"
                           f"Elapsed Time: {elapsed_training_time // 60} min {elapsed_training_time % 60:.2f} sec\n"
                           f"Patience Count : {patience_count}/{patience}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = f"Elapsed Time: {elapsed_time // 60} min {elapsed_time % 60} sec"
    print(elapsed_time_str)

    with open(model_path + 'numbers.txt', "w") as file:
        file.write(f"Top Accuracy Train Epoch : {Top_Accuracy_Train_Epoch} Accuracy : {Top_Accuracy_Train}\n"
                   f"Top Accuracy Validation Epoch : {Top_Accuracy_Validation_Epoch} Accuracy : {Top_Accuracy_Validation}\n"
                   f"Bottom Loss Train Epoch : {Bottom_Loss_Train_Epoch} Loss : {Bottom_Loss_Train}\n"
                   f"Bottom Loss Validation Epoch : {Bottom_Loss_Validation_Epoch} Loss : {Bottom_Loss_Validation}\n"
                   f"Elapsed Time: {elapsed_time // 60} min {elapsed_time % 60:.2f} sec\n"
                   f"Patience Count : {patience_count}/{patience}\n")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    with open(model_path + 'numbers.txt', "a") as file:
        file.write(f"Test Accuracy: {test_accuracy:.4f}\n")