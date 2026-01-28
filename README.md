# Customer Churn Analysis – Machine Learning Fundamentals

## 📌 Mục tiêu

Dự án này được thực hiện với mục tiêu **học và thực hành nền tảng Machine Learning**, không tập trung vào việc tối ưu model hay đạt kết quả cao nhất.

Project giúp làm quen với:
- Python cho Machine Learning
- Pandas & NumPy
- Quy trình Machine Learning cơ bản
- Cách tổ chức code ML thành một project rõ ràng

---

## 📊 Dataset

Sử dụng dataset **Telco Customer Churn**.

- Mỗi dòng là một khách hàng
- Label cần dự đoán: **Churn (Yes / No)**
- Dữ liệu bao gồm cả:
  - Numerical features
  - Categorical features

---

## 🔄 Workflow

Quy trình Machine Learning trong project:

```
Load Data → Preprocess → Train/Test Split → Train Model → Predict → Evaluate
```

---

## 🧠 Model

- Logistic Regression (baseline model)
- Mục đích: hiểu workflow, không tối ưu hyperparameter

---

## 📈 Evaluation

Các chỉ số đánh giá được sử dụng:
- Accuracy
- Precision
- Recall
- F1-score

**Kết quả:**
- Accuracy: 79.35%
- Precision (Churn): 63%
- Recall (Churn): 54%
- F1-score (Churn): 58%

---

## 📁 Cấu trúc project

```
churn_project/
├── data/
│   └── churn.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── venv/
├── README.md
└── .gitignore
```

---

## ▶️ Cách chạy project

### 1. Tạo và kích hoạt virtual environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# hoặc
source venv/bin/activate  # Linux/Mac
```

### 2. Cài đặt dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Chạy training

Từ thư mục gốc của project:

```bash
cd src
python train.py
```

Hoặc chạy trực tiếp với đường dẫn Python từ venv:

```bash
# Windows (từ thư mục gốc project)
cd src; ..\venv\Scripts\python.exe train.py
```

---

## 📝 Ghi chú

Đây là project học tập, được sử dụng để xây dựng nền tảng cho các bước tiếp theo trong Machine Learning.
