import torch
import torch.nn as nn
import torch.optim as optim

# 1) 데이터 준비
# x: -3 ~ 3 사이 값
x_train = torch.linspace(-3, 3, steps=50).unsqueeze(1)
y_train = x_train**2   # 정답: y = x^2

# 2) MLP 모델 정의 (은닉층 1개)
model = nn.Sequential(
    nn.Linear(1, 10),   # 입력 1 → 은닉층 10
    nn.ReLU(),          # 활성화 함수(ReLU)
    nn.Linear(10, 1)    # 은닉층 → 출력 1
)

# 3) 손실함수 + 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4) 학습 loop
epochs = 2000
for epoch in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5) 테스트
test = torch.tensor([[2.0]])
print("입력 2일 때 예측값:", model(test).item())
