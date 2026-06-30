# 🛍️ Retail Customer Retention AI Portal

Aplikasi pendukung keputusan berbasis **Machine Learning** yang dirancang untuk membantu perusahaan retail mengidentifikasi pelanggan yang berisiko berhenti bertransaksi (*customer churn*). Proyek ini menggunakan algoritma **XGBoost Classifier** yang dievaluasi menggunakan metode **Walk-Forward Validation (Sliding Window)** untuk memastikan prediksi pada data deret waktu dilakukan tanpa *data leakage* (*zero data leakage*).

---

## Important Links

- 🌐 **Web Application (Streamlit):**  
  https://retailchurn.streamlit.app/

- 📓 **Model Training Notebook (Google Colab):**  
  https://colab.research.google.com/drive/19fIA_rcTVONrM8EwM10-rtKdl4p2eQeO?usp=sharing

- 📓 **Model EDA Notebook (Google Colab):**  
  https://colab.research.google.com/drive/1jQfVruajW1mMIFi1AiGvld_89E4bbJ3a?usp=sharing

---

## 📂 Repository Structure

```text
.
├── app.py                  # Main Streamlit dashboard
├── ML__project_code_train.ipynb   # Model training script
├── ML_PROJECT_EDA.ipynb    # EDA script
├── model_3_months.pkl      # XGBoost model (3-month window)
├── model_6_months.pkl      # XGBoost model (6-month window)
├── model_12_months.pkl     # XGBoost model (12-month window - Best Model)
├── logo.png                # Dashboard logo
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ✨ Key Features

### 🔍 Per-Customer Prediction

Predict churn probability for an individual customer by entering **RFMT (Recency, Frequency, Monetary, Time)** features. The application instantly generates:

- Churn probability score
- Customer status prediction
- Risk interpretation

---

### 📊 Batch Processing & Executive Analytics

Upload a transaction dataset (`.csv`) to analyze multiple customers simultaneously.

The system automatically:

- Performs feature engineering
- Predicts customer churn
- Identifies at-risk customers
- Estimates potential revenue at risk
- Displays interactive analytics and summary metrics

---

### Multiple Observation Windows

Switch between different AI observation windows:

- **3 Months**
- **6 Months**
- **12 Months (Best Performing Model)**

This allows users to compare how different historical time windows influence prediction performance.

---

## Local Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/winsont123/retailchurn.git
cd retailchurn
```

---

### 2. Install Dependencies

It is recommended to use a virtual environment.

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Machine Learning Pipeline

1. Data preprocessing
2. RFMT feature engineering
3. Walk-Forward Validation (Sliding Window)
4. XGBoost model training
5. Model evaluation
6. Streamlit deployment

---

## Model

- **Algorithm:** XGBoost Classifier
- **Validation Strategy:** Walk-Forward Validation
- **Feature Engineering:** RFMT
- **Prediction Task:** Binary Customer Churn Classification

---

## Requirements

- Python 3.10+
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Plotly

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Authors

Developed as the final project for the **Machine Learning** course at **BINUS University**.

---

## 📄 License

This project is intended for educational purposes.
