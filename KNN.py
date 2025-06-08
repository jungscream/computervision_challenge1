import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# === 1. 피처 로딩 ===
df_law = pd.read_csv("law_features.csv")
df_lbp = pd.read_csv("lbp_features.csv")
df_glcm = pd.read_csv("glcm_features.csv")
df_sift = pd.read_csv("sift_features.csv")

# 라벨 기준 정렬
for df in [df_law, df_lbp, df_glcm, df_sift]:
    df.sort_values(by='label', inplace=True)
    df.reset_index(drop=True, inplace=True)

# === 2. 피처 병합 및 분할 ===
X = pd.concat([
    df_law.drop(columns='label'),
    df_lbp.drop(columns='label'),
    df_glcm.drop(columns='label'),
    df_sift.drop(columns='label')
], axis=1)
y = df_law['label']

# === 3. 정규화 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Cross-Validation (Stratified 5-Fold) ===
print("📊 [K-Fold] 교차검증 시작...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=9)

cv_scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
print("✅ Cross-Validation 정확도 (각 Fold):", np.round(cv_scores, 4))
print("✅ Cross-Validation 평균 정확도:", round(cv_scores.mean(), 4))

# === 5. 최종 Hold-out 테스트 평가 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\n📌 [Hold-Out] 테스트셋 평가 결과:")
print(classification_report(y_test, y_pred))
print("정확도:", accuracy_score(y_test, y_pred))
