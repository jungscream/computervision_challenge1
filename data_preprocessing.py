import os
import cv2
import numpy as np
from tqdm import tqdm

input_root = './testset'
output_root = './preprocessed/test'
os.makedirs(output_root, exist_ok=True)

def preprocess_stretch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stretched = cv2.equalizeHist(gray)
    return stretched

classes = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

for cls in classes:
    input_class_dir = os.path.join(input_root, cls)
    output_class_dir = os.path.join(output_root, cls)
    os.makedirs(output_class_dir, exist_ok=True)

    images = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in tqdm(images, desc=f'Processing {cls}'):
        img_path = os.path.join(input_class_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        processed = preprocess_stretch(image)
        save_path = os.path.join(output_class_dir, os.path.splitext(img_name)[0] + '.png')
        cv2.imwrite(save_path, processed)

print("✅ 전처리(stretch) 완료 → ./preprocessed/stretch")
