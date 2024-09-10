import torch
import matplotlib.pyplot as plt
from scheduler import CosineAnnealingWarmUpRestarts
import transformers

lr = 1e-3
epochs = 300
# 가상의 옵티마이저 생성
optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))], lr=lr)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=1000, steps_per_epoch=1000)
# CosineAnnealingWarmRestarts 스케줄러 설정
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=lr / 100)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
# scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=1e-5, T_up=10, gamma=0.5)
# scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
# 학습률을 저장할 리스트 초기화
lr_rates = []
for epoch in range(epochs):
    # 현재 학습률 저장
    lr_rates.append(optimizer.param_groups[0]["lr"])
    # 스케줄러 스텝
    scheduler.step()

print(lr_rates)
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), lr_rates, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("OneCycleLR Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.savefig("lr_plot.png")
plt.show()
