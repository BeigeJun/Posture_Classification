import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력 사용
        return out


# Custom Dataset 클래스
class PoseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.data['Final_Label'] = self.label_encoder.fit_transform(self.data['Final_Label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = list(map(int, self.data.iloc[idx]['Labels'].split(',')))
        final_label = self.data.iloc[idx]['Final_Label']

        # 시퀀스 형태로 변환
        return torch.tensor(labels, dtype=torch.float32), torch.tensor(final_label, dtype=torch.long)


# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (labels, final_label) in enumerate(train_loader):
            labels = labels.unsqueeze(-1)  # LSTM 입력을 위한 차원 추가
            labels = labels.to(device)
            final_label = final_label.to(device)

            # 순전파
            outputs = model(labels)
            loss = criterion(outputs, final_label)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.10f}')


# 파라미터 설정
input_size = 1  # 입력 차원
hidden_size = 64  # LSTM의 hidden state 크기
num_classes = 2  # Danger(1), Safe(0)
num_layers = 1  # LSTM 레이어 수
num_epochs = 10000  # 학습 에폭 수
learning_rate = 0.001

# 데이터 로드 및 분할
csv_file = 'pose_labels.csv'  # CSV 파일 경로
dataset = PoseDataset(csv_file)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# 모델, 손실 함수 및 옵티마이저 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, hidden_size, num_classes, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
train_model(model, train_loader, criterion, optimizer, num_epochs)

# 모델 저장
torch.save(model.state_dict(), 'lstm_pose_model.pth')
print("모델 저장 완료")
