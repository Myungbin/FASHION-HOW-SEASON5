# FASHION HOW
- 이번 대회는 모델 경량화에 초점을 맞추어 실시간 성능과 효율성을 높이는 데 중점을 둡니다.  
Sub-Task 1,2에 제공되는 FASCODE(FAShion COordination DatasEt / Fashion CODE)는 14,000 장의 패션 이미지 데이터로 구성되어 있으며, 각각의 의류 위치 정보를 나타내는 bounding box 좌표와 각각의 아이템을 세 분류의 감성 특징으로 태깅한 라벨 정보가 포함되어 있습니다. 

## Project Structure
```
FASHION-HOW 
├─ .gitignore
├─ check_points/   # save model files
├─ timm/ 
├─ augmentation.py  # data augmentaiton 
├─ config.py  # config(hyyperparameter, path)
├─ dataset.py  # dataset & dataloader
├─ draw_plot.py  # draw train result plot
├─ log.py  # environments & train log
├─ main.py 
├─ model.py  # train model
├─ remove_background.py 
├─ trainer.py  # model train & evaludation
├─ utils.py  # train utils
└─ README.md
```

## Task Experiments
   
`Subtask1`  
[Subtask-1 Information](subtask1/README.md)   
[Subtask-1 Experiments](SUB_TASK1_Experiments.pdf)  
`Subtask2`  
[Subtask-2 Information](subtask2/README.md)  
[Subtask-2 Experiments](SUB_TASK2_Experiments.pdf)


## Result
| Task      | Score  | Rank |
|-----------|--------|------|
| Sub-Task1 | 0.5056 | 2nd  |
| Sub-Task2 | 0.8667 | 2nd  |


### Development Environment
---
`Python 3.11.8`  `PyTorch 2.2.2+cu121`

```
OS: Window11
CPU: 13th Gen Intel(R) Core(TM) i5-13600KF
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
RAM: 32GB
```
## Host
주최: ETRI  
기간: 2024년 7월 29일 (월) - 9월 11일 (수)  
[Competition Link](https://fashion-how.org/)





