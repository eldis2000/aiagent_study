import torch
import torch.nn as nn

# 1. 아주 단순한 이미지 2개 (4x4)
horizontal = torch.tensor([
    [1.,1.,1.,1.],
    [0.,0.,0.,0.],
    [1.,1.,1.,1.],
    [0.,0.,0.,0.]
])

vertical = torch.tensor([
    [1.,0.,1.,0.],
    [1.,0.,1.,0.],
    [1.,0.,1.,0.],
    [1.,0.,1.,0.]
])

# CNN 입력 형태로 변환 (배치, 채널, H, W)
images = torch.stack([horizontal, vertical]).unsqueeze(1)  # (2,1,4,4)

# 2. 아주 단순한 CNN 레이어 1개만 사용
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# 3. 필터를 '가로줄 찾기' 패턴으로 직접 설정해보기
with torch.no_grad():
    conv.weight[:] = torch.tensor([[[[1.,1.,1.],
                                     [0.,0.,0.],
                                     [1.,1.,1.]]]])  # shape (1,1,3,3)

# 4. CNN 출력 확인
output = conv(images)
print("CNN 출력:\n", output)