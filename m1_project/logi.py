import torch
import torch.nn as nn
import torch.optim as optim

# 1) 데이터 준비
x_train = torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.]], dtype=torch.float32)
y_train = torch.tensor([[0.], [0.], [0.], [1.], [1.], [1.]], dtype=torch.float32)

# 2) 로지스틱 회귀 모델 정의
model = nn.Sequential(
    nn.Linear(1, 1),   # z = w*x + b
    nn.Sigmoid()       # sigmoid(z)
)

# 3) 손실 함수와 옵티마이저
criterion = nn.BCELoss()               # 이진 분류(0/1)용 loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4) 학습 루프
epochs = 1000
for epoch in range(epochs):
    y_pred = model(x_train)      # 예측 (확률)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5) 예측 테스트
test_x = torch.tensor([[2.5], [4.5], [5.5]], dtype=torch.float32)
test_pred = model(test_x)

print("\n예측 확률:")
print(test_pred)
print("\n0.5보다 크면 '합격(1)'으로 판단")
