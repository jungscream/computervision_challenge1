import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

input_dir = './preprocessed/training'
output_csv = 'lbp_features.csv'
P, R = 16, 2
METHOD = 'uniform'
LBP_BINS = P + 2

features, labels = [], []

for cls in sorted(os.listdir(input_dir)):
    cls_path = os.path.join(input_dir, cls)
    if not os.path.isdir(cls_path): continue

    for fname in tqdm(sorted(os.listdir(cls_path)), desc=f'LBP {cls}'):
        img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE)
        lbp = local_binary_pattern(img, P, R, METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_BINS + 1), range=(0, LBP_BINS))
        hist = hist.astype('float') / hist.sum()
        features.append(hist)
        labels.append(cls)

df = pd.DataFrame(features)
df['label'] = labels
df.to_csv(output_csv, index=False)
print(f"✅ LBP 저장 완료 → {output_csv}")
