# 🚀 Panduan Deployment Streamlit - Customer Churn Prediction

## 📁 File yang Diperlukan

Pastikan file-file berikut ada dalam satu folder:

```
📂 TA-11/
├── app.py                          # Aplikasi Streamlit
├── requirements.txt                # Dependencies
├── customer_churn_ann_model.h5     # Model ANN (dari notebook)
├── scaler.pkl                      # StandardScaler (dari notebook)
├── feature_columns.json            # Nama kolom fitur (dari notebook)
├── categorical_options.json        # Opsi kategorik (dari notebook)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (opsional)
```

## 🔧 Langkah-Langkah Deployment

### **Opsi 1: Jalankan Lokal**

1. **Jalankan notebook terlebih dahulu** di Google Colab untuk menghasilkan file model
2. **Download file hasil training:**
   - `customer_churn_ann_model.h5`
   - `scaler.pkl`
   - `feature_columns.json`
   - `categorical_options.json`

3. **Letakkan semua file** di folder yang sama dengan `app.py`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

6. **Buka browser** di `http://localhost:8501`

---

### **Opsi 2: Deploy ke Streamlit Cloud (GRATIS)**

1. **Buat repository GitHub baru**

2. **Upload semua file** ke repository:
   - `app.py`
   - `requirements.txt`
   - `customer_churn_ann_model.h5`
   - `scaler.pkl`
   - `feature_columns.json`
   - `categorical_options.json`

3. **Buka [Streamlit Cloud](https://share.streamlit.io/)**

4. **Klik "New app"**

5. **Connect GitHub repository**

6. **Pilih:**
   - Repository: `<nama-repo-anda>`
   - Branch: `main`
   - Main file path: `app.py`

7. **Klik "Deploy!"**

8. **Tunggu proses deployment** (sekitar 2-5 menit)

9. **Aplikasi akan live** di URL seperti:
   `https://<nama-repo>-<username>.streamlit.app`

---

## ⚠️ Catatan Penting

### Untuk Google Colab:
Setelah menjalankan notebook, download file-file yang dihasilkan:
```python
# Di Colab, gunakan ini untuk download
from google.colab import files

files.download('customer_churn_ann_model.h5')
files.download('scaler.pkl')
files.download('feature_columns.json')
files.download('categorical_options.json')
```

### Ukuran File Model:
- Jika file model terlalu besar untuk GitHub (>100MB), gunakan **Git LFS** atau upload ke Google Drive/cloud storage lain

### Versi TensorFlow:
- Pastikan versi TensorFlow saat training sama dengan saat deployment
- Jika ada error, coba sesuaikan versi di `requirements.txt`

---

## 🎯 Fitur Aplikasi

1. **Input Data Pelanggan** - Form lengkap untuk semua fitur
2. **Prediksi Real-time** - Hasil prediksi instan
3. **Visualisasi Gauge** - Tampilan probabilitas churn
4. **Rekomendasi Tindakan** - Saran berdasarkan hasil prediksi
5. **Responsive Design** - Tampilan optimal di berbagai perangkat

---

## 📞 Troubleshooting

| Error | Solusi |
|-------|--------|
| Model not found | Pastikan file `.h5` ada di folder yang sama |
| Import error | Jalankan `pip install -r requirements.txt` |
| Memory error | Coba reduce model size atau gunakan server lebih besar |
| Version mismatch | Sesuaikan versi di requirements.txt |

---

**Happy Deploying! 🚀**
