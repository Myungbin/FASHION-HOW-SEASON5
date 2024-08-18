import os
from rembg import remove
import cv2
from tqdm import tqdm

# 원본 이미지 폴더 경로
input_folder = r"C:\workspace\dataset\FashionHow\subtask2\val"
# 처리된 이미지를 저장할 폴더 경로
output_folder = r"C:\workspace\dataset\FashionHow\subtask2\val2"


# output_folder가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# input_folder 내의 모든 폴더와 파일을 순회
for label in os.listdir(input_folder):
    label_path = os.path.join(input_folder, label)
    
    # 폴더일 경우만 처리
    if os.path.isdir(label_path):
        output_label_path = os.path.join(output_folder, label)
        
        # output_folder 내에 동일한 라벨 폴더가 없으면 생성
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)
        
        # 라벨 폴더 내의 모든 이미지 파일을 순회
        for image_file in tqdm(os.listdir(label_path)):
            input_image_path = os.path.join(label_path, image_file)
            output_image_path = os.path.join(output_label_path, image_file)
            
            input_image = cv2.imread(input_image_path)

            output_image = remove(input_image)
            output_image = output_image[:, :, :3]

            # 배경 제거된 이미지를 저장 (PNG 포맷으로 저장)
            cv2.imwrite(output_image_path, output_image)

print("모든 이미지의 배경 제거 및 저장이 완료되었습니다!")
