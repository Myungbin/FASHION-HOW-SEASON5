import torch
import matplotlib.pyplot as plt
from scheduler import CosineAnnealingWarmUpRestarts
lr = 1e-3
# 가상의 옵티마이저 생성
optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))], lr=lr)

# CosineAnnealingWarmRestarts 스케줄러 설정
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=lr / 100)
# scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=1e-5, T_up=10, gamma=0.5)
# 학습률을 저장할 리스트 초기화
lr_rates = []

# 총 100 에포크에 걸쳐 스케줄러의 학습률 변화를 기록
epochs = 300
for epoch in range(epochs):
    # 현재 학습률 저장
    lr_rates.append(optimizer.param_groups[0]['lr'])
    # 스케줄러 스텝
    scheduler.step()

print(lr_rates)
# 학습률 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), lr_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('CosineAnnealingWarmRestarts Learning Rate Schedule')
plt.legend()
plt.grid(True)
plt.show()
