import os
import cv2
import numpy as np
from tqdm import tqdm

input_dir = './testset'
output_dir = './preprocessed/test'
os.makedirs(output_dir, exist_ok=True)

def preprocess_stretch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stretched = cv2.equalizeHist(gray)
    return stretched

images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(sorted(images), desc='Processing test images'):
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    processed = preprocess_stretch(image)
    save_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.png')
    cv2.imwrite(save_path, processed)

print("✅ 전처리(stretch) 완료 → ./preprocessed/test")
