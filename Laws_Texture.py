from scipy.signal import convolve2d
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

law_kernels = [
    np.outer(a, b) for a in [[1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, -1]]
    for b in [[1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, -1]]
]

input_dir = './preprocessed/training'
output_csv = 'law_features.csv'
features, labels = [], []

for cls in sorted(os.listdir(input_dir)):
    cls_path = os.path.join(input_dir, cls)
    if not os.path.isdir(cls_path): continue

    for fname in tqdm(sorted(os.listdir(cls_path)), desc=f'Law {cls}'):
        img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        feat_vec = []
        for k in law_kernels:
            filtered = convolve2d(img, k, mode='same', boundary='symm')
            feat_vec.append(np.mean(np.abs(filtered)))
        features.append(feat_vec)
        labels.append(cls)

df = pd.DataFrame(features)
df['label'] = labels
df.to_csv(output_csv, index=False)
print(f"✅ Law 저장 완료 → {output_csv}")
