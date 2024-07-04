import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# CSV 파일 경로
csv_file_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Angle_data.csv'

# CSV 파일 읽기
data = pd.read_csv(csv_file_path)

# 피처와 라벨 분리
X = data.drop('label', axis=1).values  # 피처 (각 keypoint 간 각도)
y = data['label'].values  # 라벨

# 라벨 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 트레인셋과 테스트셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 데이터 로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 신경망 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 모델 초기화
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = MLP(input_size, num_classes)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10000
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 저장
model_path = 'model_MLP.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 저장된 모델 불러오기
loaded_model = MLP(input_size, num_classes)
loaded_model.load_state_dict(torch.load(model_path))
print("Model loaded from", model_path)

# 모델 평가
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
