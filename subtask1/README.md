# SubTask1
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
1. Remove Background
    - 이미지 Background를 지운 train / val 이미지 저장 (remove_background.py)
2. Main
    - config.py 에서 root & df path 경로에 맞게 수정
    - main.py 실행
```

## Experiments
Subtask1은 모델 크기 제한(25M)이 있었음. Pretrain 모델은 사용하능하기 때문에 [timm](https://github.com/huggingface/pytorch-image-models)에서 25MB제한 내에서 성능이 좋았던 Pretrain 모델을 가져와 사용하였음.  
Convolution 기반 모델인 ConvNeXt와 MobileNet에 비해서 ViT기반 모델이 성능이 좋았음

### **Train Info**

**Base Train Environments**

EVA02 Tiny + CrossEntropy + AdamW + CosineAnnealingWarmRestarts  
Albumentation Augmentation  

**Augmentation**

- 다양한 Augmentation이 효과가 있었음. 하지만 기본으로 쓰는 Augmentation보다 성능이 좋은경우는 없었음 (기본 Augmentation에 추가하는 경우 효과가 미미)
- 이미지를 확률적으로 Crop하여 데이터에 다양성을 주는 테스트도 하였지만 효과가 크지 않았음. 
- Remove Background를 단일로 했을 때 유의미한 성능 향상이 있었음.
- Remove Background + Raw Data를 한번에 학습하는 테스트도 하였지만, 효과가 크지 않고 기존 Rembg 데이터만 가지고 학습했을 때 보다 성능이 좋지 않았음.
- Cutmix를 할 경우 학습시간이 크게 증가하지만 유의미한 성능 향상이 있지 않았음. 각 Label이 동시에 Cutmix가 들어가기 때문에 지나친 데이터 복잡도를 유발했다고 생각함.

**Etc**

- Test Score는 Validation Acc를 대부분 정직하게 따라갔음.
- Validation Acc가 0.65를 넘는지에 대한 여부를 기준으로 모델링을 진행하였음
- Asymmetric Loss와 Focal Loss의 경우 성능이 좋지 않았음.
- label smoothing의 영향이 생각보다 컸음

