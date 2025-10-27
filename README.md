#  Kira / Ev Fiyat Tahmin Projesi (House Price Prediction)

Bu proje, **Kaggle "House Prices - Advanced Regression Techniques"** veri seti kullanılarak hazırlanmıştır.  
Amaç, evin çeşitli özelliklerine göre satış fiyatını tahmin eden **makine öğrenmesi modelleri** geliştirmektir.

---

##  Veri Seti Bilgisi
- **Kaynak:** [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Açıklama:**  
  Veri seti, konutların fiziksel ve konumsal özelliklerine ait 80’den fazla değişken içerir.  
  Hedef değişken (**SalePrice**) evin satış fiyatını temsil eder.  
- **Dosya:** `train.csv`

---

##  Proje Özellikleri
-  **Veri Temizleme:** Eksik değerlerin median veya 'None' ile doldurulması  
-  **Veri Görselleştirme:** Seaborn ile dağılım ve korelasyon analizleri  
-  **Modelleme:**  
  - Linear Regression  
  - Random Forest Regressor  
  - XGBoost Regressor  
-  **Model Değerlendirmesi:** RMSE (Root Mean Squared Error) ve MAPE (Mean Absolute Percentage Error)  
-  **Özellik Önemi Analizi:** Random Forest ile feature importance grafiği  

---

##  Kullanılan Teknolojiler
- **Python 3.9+**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Scikit-Learn**
- **XGBoost**

---

##  Kurulum ve Çalıştırma

### 1️. Gerekli kütüphaneleri yükle
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2️. Projeyi çalıştır
```bash
python kira_tahmin.py
```

## Model Karşılaştırması

| Model              | Ortalama RMSE  | RMSE Std. Sapması| MAPE (%) |
|--------------------|----------------|------------------|-----------|
| Linear Regression  | ~              | ~                | ~         |
| Random Forest      | ~              | ~                | ~         |
| XGBoost            | ~              | ~                | ~         |


##  Dosya Yapısı
```
kira-tahmin
 ┣  kira_tahmin.py         # Modelleme ve analiz kodları
 ┣  train.csv              # Kaggle veri seti
 ┗  README.md
```

Developer: Sedanur Peker
