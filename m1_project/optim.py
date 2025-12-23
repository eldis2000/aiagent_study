import torch
import torch.nn as nn
import torch.optim as optim

# 1. 학습 데이터 (텐서)
x_train = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float32)
y_train = torch.tensor([[3.], [5.], [7.], [9.]], dtype=torch.float32)  # 실제 공식: y=2x+1

# 2. 모델 정의 (선형 1층)
model = nn.Linear(1, 1)

# 3. 손실 함수와 옵티마이저 설정
criterion = nn.MSELoss()              # 평균제곱오차
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 경사하강법

# 4. 학습 루프
epochs = 1000
for epoch in range(epochs):
    # 순전파
    y_pred = model(x_train)

    # 손실 계산
    loss = criterion(y_pred, y_train)

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. 테스트
test_val = torch.tensor([[5.]], dtype=torch.float32)
prediction = model(test_val).item()
print("5 입력했을 때 예측:", prediction)
