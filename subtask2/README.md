# SubTask2
### Development Environment
`Python 3.11.8`  `PyTorch 2.2.2+cu121`

```
OS: Window11
CPU: 13th Gen Intel(R) Core(TM) i5-13600KF
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
RAM: 32GB
```

## Process
```
1. Main
    - config.py 에서 root & df path 경로에 맞게 수정
    - main.py 실행
```

## Experiments
Subtask2는 Pretrain 불가능 + 모델크기 제한(25M)이 있었음.  
ViT 기반 모델은 Conv 모델에 비해 더 많은 데이터셋이 있어야 성능이 좋다고 알려져 있지만, Subtask2에서는 사전 학습이 없어도, ViT기반 모델이 더 성능이 좋았음. (비교모델: ConvNeXt, MobileNet)

### **Train Info**

**Base Train Environments**

EVA02 Tiny + CrossEntropy + AdamW + CosineAnnealingWarmRestarts  
Albumentation Augmentation  

**Augmentation**

- SubTask2는 이미지 색을 맞추는 Task기 때문에 Augmentation은 색의 변화를 주지 않는 Augmentation만 적용하였음. 
- HSV Transform은 성능 향상에 좋지 않았음
- 이미지를 확률적으로 Crop하여 해당 이미지의 색을 맞추는 테스트도 하였지만 효과가 크지 않았음.
- CutMix + Mixup을 적용해봤지만, 성능 향상에 좋지 않았음 => 색을 판단해야 하는데 Mixup은 데이터에 노이즈만 추가한 것이라 생각
- Remove Background + Raw Data를 한번에 학습하는 테스트도 하였지만, 효과가 크지 않았음.

**Etc**

- Test Score는 Validation Acc를 대부분 정직하게 따라갔음.
- Asymmetric Loss와 Focal Loss의 경우 성능이 좋지 않았음.


