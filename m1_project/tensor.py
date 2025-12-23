import torch# 텐서 생성 및 기본 연산

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

y = torch.ones_like(x)

z = x + y

print(z)