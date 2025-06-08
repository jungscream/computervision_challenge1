import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import csv

# === 1. 학습용 피처 로딩 ===
df_law = pd.read_csv("law_features.csv")
df_lbp = pd.read_csv("lbp_features.csv")
df_glcm = pd.read_csv("glcm_features.csv")
df_sift = pd.read_csv("sift_features.csv")

# 컬럼 이름 지정
df_law.columns = [f'law_{i}' for i in range(df_law.shape[1] - 1)] + ['label']
df_lbp.columns = [f'lbp_{i}' for i in range(df_lbp.shape[1] - 1)] + ['label']
df_glcm.columns = ['glcm_contrast', 'glcm_energy', 'glcm_homogeneity', 'glcm_correlation', 'label']
df_sift.columns = [f'sift_{i}' for i in range(df_sift.shape[1] - 1)] + ['label']

# 학습 데이터 병합 및 정규화
X_train = pd.concat([
    df_law.drop(columns='label'),
    df_lbp.drop(columns='label'),
    df_glcm.drop(columns='label'),
    df_sift.drop(columns='label')
], axis=1)
y_train = df_law['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# KNN 학습
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled, y_train)

# === 2. 테스트용 피처 로딩 (filename 유지) ===
test_law = pd.read_csv("challenge_features_LAW.csv").drop(columns='label', errors='ignore')
test_lbp = pd.read_csv("challenge_features_LBP.csv").drop(columns='label', errors='ignore')
test_glcm = pd.read_csv("challenge_features_GLCM.csv").drop(columns='label', errors='ignore')
test_sift = pd.read_csv("challenge_features_SIFT.csv").drop(columns='label', errors='ignore')

# filename 분리
filenames = test_lbp['filename']  # 어느 파일에서든 동일하다고 가정
test_law.drop(columns='filename', inplace=True, errors='ignore')
test_lbp.drop(columns='filename', inplace=True, errors='ignore')
test_glcm.drop(columns='filename', inplace=True, errors='ignore')
test_sift.drop(columns='filename', inplace=True, errors='ignore')

# 컬럼 이름 정렬
test_law.columns = [f'law_{i}' for i in range(test_law.shape[1])]
test_lbp.columns = [f'lbp_{i}' for i in range(test_lbp.shape[1])]
test_glcm.columns = ['glcm_contrast', 'glcm_energy', 'glcm_homogeneity', 'glcm_correlation']
test_sift.columns = [f'sift_{i}' for i in range(test_sift.shape[1])]

# 병합 및 정규화
X_test = pd.concat([test_law, test_lbp, test_glcm, test_sift], axis=1)
X_test_scaled = scaler.transform(X_test)

# === 3. 예측 ===
predicted_labels = knn.predict(X_test_scaled)

# === 4. 결과 저장 (filename 기반) ===
with open('c1_t1_a1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for filename, label in zip(filenames, predicted_labels):
        writer.writerow([filename, label])

print("✅ 예측 완료 → c1_t1_a1.csv 생성됨")
