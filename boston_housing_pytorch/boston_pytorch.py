# =========================================
# Boston Housing Price Prediction (PyTorch)
# Single File Version
# =========================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -----------------------------------------``
# 1. Reproducibility
# -----------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------
# 2. Load Boston Housing Dataset (OpenML)
# -----------------------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

X = df.drop(columns=["MEDV"]).values   # features (13)
y = df["MEDV"].values.reshape(-1, 1)   # target

# -----------------------------------------
# 3. Train / Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 4. Standard Scaling
# -----------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------
# 5. Convert to Torch Tensor
# -----------------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------------------
# 6. Model Definition
# -----------------------------------------
class BostonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = BostonModel()

# -----------------------------------------
# 7. Loss & Optimizer
# -----------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------------------
# 8. Training Loop
# -----------------------------------------
epochs = 500

for epoch in range(epochs):
    model.train()

    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"[Epoch {epoch+1:3d}/{epochs}] Loss: {loss.item():.4f}")

# -----------------------------------------
# 9. Evaluation
# -----------------------------------------
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    mse = mean_squared_error(y_test.numpy(), test_preds.numpy())
    rmse = np.sqrt(mse)

print("\n===== Evaluation Result =====")
print(f"RMSE: {rmse:.3f}")

# -----------------------------------------
# 10. Prediction Sample
# -----------------------------------------
print("\n===== Sample Predictions =====")
for i in range(5):
    actual = y_test[i].item()
    predicted = test_preds[i].item()
    print(f"Actual: {actual:5.1f} | Predicted: {predicted:5.1f}")
