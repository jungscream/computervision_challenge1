from scipy.signal import convolve2d
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Law's Kernel 정의
law_kernels = [
    np.outer(a, b) for a in [[1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, -1]]
    for b in [[1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, -1]]
]

input_dir = './preprocessed/test'
output_csv = 'challenge_features_LAW.csv'
features = []
filenames = []

for fname in tqdm(sorted(os.listdir(input_dir)), desc='Law for challenge images'):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = img.astype(np.float32)

    feat_vec = []
    for k in law_kernels:
        filtered = convolve2d(img, k, mode='same', boundary='symm')
        feat_vec.append(np.mean(np.abs(filtered)))

    features.append(feat_vec)
    filenames.append(fname)

df = pd.DataFrame(features, columns=[f'law_{i}' for i in range(len(law_kernels))])
df['filename'] = filenames
df.to_csv(output_csv, index=False)
print(f"✅ Law 저장 완료 → {output_csv}")
