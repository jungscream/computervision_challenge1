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
X_train = pd.concat([df_law.drop(columns='label'),
                     df_lbp.drop(columns='label'),
                     df_glcm.drop(columns='label'),
                     df_sift.drop(columns='label')], axis=1)
y_train = df_law['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# KNN 학습 (이웃 수 = 10)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled, y_train)

# === 2. 테스트 피처 로딩 및 정규화 ===
def load_and_prepare_test_features():
    test_law = pd.read_csv("challenge_features_LAW.csv").drop(columns='label', errors='ignore')
    test_lbp = pd.read_csv("challenge_features_LBP.csv").drop(columns='label', errors='ignore')
    test_glcm = pd.read_csv("challenge_features_GLCM.csv").drop(columns='label', errors='ignore')
    test_sift = pd.read_csv("challenge_features_SIFT.csv").drop(columns='label', errors='ignore')

    filenames = test_lbp['filename']  # 기준 컬럼
    test_law.drop(columns='filename', inplace=True, errors='ignore')
    test_lbp.drop(columns='filename', inplace=True, errors='ignore')
    test_glcm.drop(columns='filename', inplace=True, errors='ignore')
    test_sift.drop(columns='filename', inplace=True, errors='ignore')

    test_law.columns = [f'law_{i}' for i in range(test_law.shape[1])]
    test_lbp.columns = [f'lbp_{i}' for i in range(test_lbp.shape[1])]
    test_glcm.columns = ['glcm_contrast', 'glcm_energy', 'glcm_homogeneity', 'glcm_correlation']
    test_sift.columns = [f'sift_{i}' for i in range(test_sift.shape[1])]

    X_test = pd.concat([test_law, test_lbp, test_glcm, test_sift], axis=1)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, filenames

X_test_scaled, filenames = load_and_prepare_test_features()

# === 3. Top-10 Retrieval 예측 ===
neighbors_indices = knn.kneighbors(X_test_scaled, return_distance=False)
top10_preds = np.array([[y_train.iloc[idx] for idx in row] for row in neighbors_indices])

# === 4. CSV 저장 (Top-10 라벨 저장)
with open('c1_t2_a1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for fname, preds in zip(filenames, top10_preds):
        writer.writerow([fname] + list(preds))

print("✅ Top-10 Retrieval 결과 저장 완료 → c1_t2_a1.csv")
