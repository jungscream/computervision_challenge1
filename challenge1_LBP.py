import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

input_dir = './preprocessed/test'
output_csv = 'challenge_features_LBP.csv'
P, R = 16, 2
METHOD = 'uniform'
LBP_BINS = P + 2

features = []
filenames = []

# query01.png, query02.png 등 순서대로
for fname in tqdm(sorted(os.listdir(input_dir)), desc='LBP for challenge images'):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    lbp = local_binary_pattern(img, P, R, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_BINS + 1), range=(0, LBP_BINS))
    hist = hist.astype('float') / hist.sum()

    features.append(hist)
    filenames.append(fname)

# 저장
df = pd.DataFrame(features, columns=[f'lbp_{i}' for i in range(LBP_BINS)])
df['filename'] = filenames
df.to_csv(output_csv, index=False)
print(f"✅ LBP 저장 완료 → {output_csv}")
