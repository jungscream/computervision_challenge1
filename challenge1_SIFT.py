import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

input_dir = './preprocessed/test'
output_csv = 'challenge_features_SIFT.csv'

sift = cv2.SIFT_create()
features = []
filenames = []

for fname in tqdm(sorted(os.listdir(input_dir)), desc='SIFT for challenge images'):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    kp, des = sift.detectAndCompute(img, None)
    if des is not None and len(des) > 0:
        descriptor_mean = np.mean(des, axis=0)
    else:
        descriptor_mean = np.zeros(128)

    features.append(descriptor_mean)
    filenames.append(fname)

df = pd.DataFrame(features, columns=[f'sift_{i}' for i in range(128)])
df['filename'] = filenames
df.to_csv(output_csv, index=False)
print(f"✅ SIFT 저장 완료 → {output_csv}")
