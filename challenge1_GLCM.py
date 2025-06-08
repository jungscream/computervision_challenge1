from skimage.feature import graycomatrix, graycoprops
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

input_dir = './preprocessed/test'
output_csv = 'challenge_features_GLCM.csv'
distances = [1, 2]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
levels = 256

features = []
filenames = []

for fname in tqdm(sorted(os.listdir(input_dir)), desc='GLCM for challenge images'):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    glcm = graycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feats = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    features.append(feats)
    filenames.append(fname)

df = pd.DataFrame(features, columns=['contrast', 'energy', 'homogeneity', 'correlation'])
df['filename'] = filenames
df.to_csv(output_csv, index=False)
print(f"✅ GLCM 저장 완료 → {output_csv}")
