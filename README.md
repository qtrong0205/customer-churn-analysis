# 🎯 Dự đoán Customer Churn - Học Pipeline Machine Learning

## 📌 Mục đích dự án

Dự án này được xây dựng với mục đích **học và thực hành quy trình (pipeline) Machine Learning hoàn chỉnh** từ đầu đến cuối, bao gồm:

- Thu thập và khám phá dữ liệu (EDA)
- Tiền xử lý dữ liệu (Preprocessing)
- Xây dựng và huấn luyện mô hình (Training)
- Tinh chỉnh siêu tham số (Hyperparameter Tuning)
- Đánh giá mô hình (Evaluation)
- Dự đoán trên dữ liệu mới (Prediction)

> **Bài toán:** Dự đoán khách hàng có rời bỏ dịch vụ (churn) hay không dựa trên các đặc điểm của họ.

---

## 📁 Cấu trúc dự án

```
churn-ml/
├── data/
│   └── raw/
│       └── churn.csv          # Dữ liệu gốc
├── notebooks/
│   └── eda.ipynb              # Phân tích khám phá dữ liệu
├── src/
│   ├── preprocess.py          # Tiền xử lý dữ liệu
│   ├── train.py               # Huấn luyện mô hình
│   ├── tuning.py              # Tinh chỉnh siêu tham số
│   ├── evaluate.py            # Đánh giá mô hình
│   ├── metrics.py             # Các metrics đánh giá
│   └── predict.py             # Dự đoán
├── requirements.txt           # Thư viện cần thiết
└── README.md                  # Tài liệu dự án
```

---

## 🔄 Pipeline Machine Learning

### 1️⃣ Khám phá dữ liệu (EDA)
- Phân tích phân phối các features
- Kiểm tra missing values
- Tìm hiểu mối tương quan giữa các biến

### 2️⃣ Tiền xử lý dữ liệu
- Xử lý missing values
- Encode categorical features (One-Hot Encoding)
- Scale numerical features (StandardScaler)
- Chia dữ liệu: Train (60%) / Validation (20%) / Test (20%)

### 3️⃣ Huấn luyện mô hình
- Logistic Regression
- Random Forest Classifier

### 4️⃣ Tinh chỉnh siêu tham số
- GridSearchCV với Cross-Validation
- Tìm bộ tham số tối ưu

### 5️⃣ Đánh giá mô hình
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Classification Report

---

## 🚀 Hướng dẫn sử dụng

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Chạy tiền xử lý
```bash
cd src
python preprocess.py
```

### Huấn luyện mô hình
```bash
python train.py
```

### Tinh chỉnh siêu tham số
```bash
python tuning.py
```

---

## 📊 Dataset

Sử dụng **Telco Customer Churn Dataset** với các features:
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, InternetService, OnlineSecurity, ...
- **Account:** Contract, PaymentMethod, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes/No)

---

## 🛠️ Công nghệ sử dụng

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## 📚 Kiến thức học được

✅ Quy trình ML end-to-end  
✅ Xử lý dữ liệu thực tế (missing values, encoding, scaling)  
✅ Train/Validation/Test split đúng cách  
✅ Tránh data leakage  
✅ Cross-validation và GridSearch  
✅ Đánh giá mô hình với nhiều metrics  

---

## 👤 Tác giả

**qtrong0205**

---

*Dự án này phục vụ mục đích học tập về Machine Learning Pipeline.*
