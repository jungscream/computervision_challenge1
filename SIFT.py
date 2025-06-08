import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

input_dir = './preprocessed/training'

sift = cv2.SIFT_create()
output_csv = 'sift_features.csv'
features, labels = [], []

for cls in sorted(os.listdir(input_dir)):
    cls_path = os.path.join(input_dir, cls)
    if not os.path.isdir(cls_path): continue

    for fname in tqdm(sorted(os.listdir(cls_path)), desc=f'SIFT {cls}'):
        img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptor_mean = np.mean(des, axis=0)
        else:
            descriptor_mean = np.zeros(128)
        features.append(descriptor_mean)
        labels.append(cls)

df = pd.DataFrame(features, columns=[f'sift_{i}' for i in range(128)])
df['label'] = labels
df.to_csv(output_csv, index=False)
print(f"✅ SIFT 저장 완료 → {output_csv}")
