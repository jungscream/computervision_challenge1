from skimage.feature import graycomatrix, graycoprops
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

input_dir = './preprocessed/training'
output_csv = 'glcm_features.csv'
distances = [1, 2]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
levels = 256
features, labels = [], []

for cls in sorted(os.listdir(input_dir)):
    cls_path = os.path.join(input_dir, cls)
    if not os.path.isdir(cls_path): continue

    for fname in tqdm(sorted(os.listdir(cls_path)), desc=f'GLCM {cls}'):
        img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE)
        glcm = graycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        feats = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0]
        ]
        features.append(feats)
        labels.append(cls)

df = pd.DataFrame(features, columns=['contrast', 'energy', 'homogeneity', 'correlation'])
df['label'] = labels
df.to_csv(output_csv, index=False)
print(f"✅ GLCM 저장 완료 → {output_csv}")
