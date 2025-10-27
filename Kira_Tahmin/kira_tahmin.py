import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb # type: ignore

# =====================================================
# Veri Seti Bilgisi
# =====================================================
# Kaggle "House Prices - Advanced Regression Techniques"
np.random.seed(42)

# --- 1. Veri Yükleme ---
print("### 1. Veri Yükleme ve İnceleme ###")
try:
    df = pd.read_csv("C:/Users/sedan/OneDrive/Belgeler/Python/Kira Tahmin/train.csv")
    print(f"Veri setinin boyutu: {df.shape}")
    print("\nSütunlar ve eksik değer sayıları:")
    print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))
except FileNotFoundError:
    print("HATA: 'train.csv' bulunamadı!")
    exit()

# --- 2. Veri Temizleme ---
print("\n### 2. Veri Temizleme ###")
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df.drop(columns=cols_to_drop, inplace=True)

categorical_cols = df.select_dtypes(include='object').columns
numerical_cols = df.select_dtypes(exclude='object').columns

for col in categorical_cols:
    df[col] = df[col].fillna('None')
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"Toplam eksik değer sayısı: {df.isnull().sum().sum()}")

# --- 3. Veri Görselleştirme ---
plt.figure(figsize=(10,6))
sns.histplot(df['SalePrice'], kde=True)
plt.title('SalePrice Dağılımı')
plt.xlabel('Fiyat')
plt.ylabel('Frekans')
plt.show()

df['SalePrice'] = np.log1p(df['SalePrice'])
print("SalePrice logaritmik dönüşüm uygulandı.")

# Korelasyon analizi
numerical_cols_after_drop = df.select_dtypes(exclude='object').columns
corr_matrix = df[numerical_cols_after_drop].corr()
top_10_corr = corr_matrix.nlargest(11, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(12,10))
sns.heatmap(df[top_10_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('En Yüksek Korelasyona Sahip Sayısal Özellikler')
plt.show()

# --- 4. Veri Ön İşleme ---
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# --- 5. Modeller ---
linear_reg_pipeline = Pipeline([('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])

random_forest_pipeline = Pipeline([('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

xgb_pipeline = Pipeline([('preprocessor', preprocessor),
                         ('regressor', xgb.XGBRegressor(objective='reg:squarederror',
                                                        n_estimators=1000,
                                                        learning_rate=0.05,
                                                        random_state=42))])

models = {
    'Linear Regression': linear_reg_pipeline,
    'Random Forest': random_forest_pipeline,
    'XGBoost': xgb_pipeline
}

# --- 6. Model Değerlendirme Fonksiyonu ---
def evaluate_model(model, X, y, cv=5):
    scores_rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(scores_rmse)
    
    # MAPE için manuel cross-validation
    mape_scores = []
    kf = cv if isinstance(cv, int) else len(cv)
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=kf, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        y_pred_cv_exp = np.expm1(y_pred_cv)
        y_test_cv_exp = np.expm1(y_test_cv)
        mape_scores.append(mean_absolute_percentage_error(y_test_cv_exp, y_pred_cv_exp))
    
    return rmse.mean(), rmse.std(), np.mean(mape_scores)

# --- 7. Tüm Modelleri Değerlendir ---
results = []
for name, model in models.items():
    mean_rmse, std_rmse, mean_mape = evaluate_model(model, X, y)
    results.append({'Model': name, 'RMSE Ortalaması': mean_rmse, 
                    'RMSE Std. Sapması': std_rmse,
                    'MAPE (%)': mean_mape*100})

results_df = pd.DataFrame(results).set_index('Model')
print("\n### Performans Raporu ###")
print(results_df)

# --- 8. Özellik Önem Grafiği (Random Forest) ---
def plot_feature_importance(pipeline, numerical_features, categorical_features, top_n=20):
    pipeline.fit(X, y)
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    feature_names = numerical_features.tolist() + list(ohe.get_feature_names_out(categorical_features))
    importances = pipeline.named_steps['regressor'].feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12,8))
    sns.barplot(x='importance', y='feature', data=fi_df.head(top_n))
    plt.title('Random Forest - En Önemli Özellikler')
    plt.tight_layout()
    plt.show()

plot_feature_importance(random_forest_pipeline, numerical_features, categorical_features)
