"""
Streamlit App - Prediksi Customer Churn dengan ANN
Praktikum Machine Learning - Pertemuan 11
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Prediksi Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

# Google Fonts & Modern CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
body, html, .main, .block-container {
    font-family: 'Poppins', Arial, sans-serif !important;
    background: radial-gradient(circle at 20% 20%, rgba(118,75,162,0.14), rgba(37,117,252,0)) ,
                radial-gradient(circle at 80% 10%, rgba(37,117,252,0.12), rgba(37,117,252,0)) ,
                linear-gradient(135deg, #f9fbff 0%, #eef3ff 45%, #f7f3ff 100%);
    background-attachment: fixed;
}
.main-header {
    background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
    padding: 2.5rem 1rem 2rem 1rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0 6px 24px rgba(80,80,180,0.13);
    position: relative;
    overflow: hidden;
}
.main-header::before, .main-header::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    filter: blur(32px);
    opacity: 0.35;
}
.main-header::before {
    width: 240px; height: 240px;
    background: #ffffff;
    top: -60px; left: -40px;
}
.main-header::after {
    width: 280px; height: 280px;
    background: #ffd166;
    bottom: -90px; right: -30px;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    letter-spacing: 1px;
}
.main-header p {
    color: rgba(255,255,255,0.93);
    font-size: 1.15rem;
    font-weight: 400;
}
.main-header .badge {
    display: inline-block;
    background: #fff;
    color: #2575fc;
    font-weight: 600;
    border-radius: 20px;
    padding: 0.3rem 1.1rem;
    font-size: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(80,80,180,0.10);
}
.info-card {
    background: rgba(255,255,255,0.92);
    padding: 1.7rem 1.3rem;
    border-radius: 13px;
    box-shadow: 0 2px 16px rgba(80,80,180,0.08);
    margin-bottom: 1.2rem;
    border-left: 5px solid #2575fc;
    transition: box-shadow 0.2s;
    backdrop-filter: blur(6px);
}
.info-card:hover {
    box-shadow: 0 6px 24px rgba(80,80,180,0.16);
}
.metric-card, .metric-card-churn, .metric-card-safe {
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(80,80,180,0.10);
    padding: 1.5rem 1rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s;
    backdrop-filter: blur(6px);
}
.metric-card:hover, .metric-card-churn:hover, .metric-card-safe:hover {
    box-shadow: 0 6px 24px rgba(80,80,180,0.18);
}
.metric-card {
    background: linear-gradient(120deg, #f6f7fb 0%, #e3e6f3 100%);
}
.metric-card-churn {
    background: linear-gradient(120deg, #ff6b6b 0%, #ee5a24 100%);
    color: #fff;
}
.metric-card-safe {
    background: linear-gradient(120deg, #26de81 0%, #20bf6b 100%);
    color: #fff;
}
.result-churn, .result-safe {
    padding: 2.2rem 1.2rem;
    border-radius: 16px;
    color: #fff;
    text-align: center;
    margin: 1.2rem 0;
    font-size: 1.2rem;
    font-weight: 600;
    box-shadow: 0 2px 12px rgba(80,80,180,0.10);
    backdrop-filter: blur(6px);
}
.result-churn {
    background: linear-gradient(120deg, #ff6b6b 0%, #ee5a24 100%);
}
.result-safe {
    background: linear-gradient(120deg, #26de81 0%, #20bf6b 100%);
}
.stButton > button {
    background: linear-gradient(120deg, #2575fc 0%, #6a11cb 100%);
    color: #fff;
    border: none;
    padding: 0.85rem 2.2rem;
    border-radius: 30px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(80,80,180,0.10);
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
    transform: translateY(-2px) scale(1.04);
    box-shadow: 0 6px 24px rgba(80,80,180,0.18);
}
.custom-divider {
    height: 3px;
    background: linear-gradient(90deg, #2575fc, #6a11cb, #2575fc);
    border: none;
    margin: 2.2rem 0 2rem 0;
    border-radius: 2px;
}
.footer {
    text-align: center;
    padding: 2.2rem 1rem 1.2rem 1rem;
    color: #666;
    border-top: 1px solid #e3e6f3;
    margin-top: 3.2rem;
    font-size: 1.05rem;
    background: #f6f7fb;
}
.footer p {
    margin-bottom: 0.2rem;
}
.tooltip-text {
    font-size: 0.93rem;
    color: #666;
    font-style: italic;
}
.streamlit-expanderHeader {
    background-color: #f0f2f6;
    border-radius: 10px;
}
.badge {
    display: inline-block;
    background: #2575fc;
    color: #fff;
    font-weight: 600;
    border-radius: 20px;
    padding: 0.3rem 1.1rem;
    font-size: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(80,80,180,0.10);
}
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & ARTIFACTS ====================
@st.cache_resource
def load_artifacts():
    """Load model dan file pendukung"""
    model = load_model('customer_churn_ann_model.h5')
    scaler = joblib.load('scaler.pkl')
    
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    with open('categorical_options.json', 'r') as f:
        categorical_options = json.load(f)
    
    # Threshold default lebih sensitif ke churn bila file tidak ada
    threshold = 0.35
    try:
        with open('threshold.json', 'r') as f:
            threshold = json.load(f).get('threshold', threshold)
    except FileNotFoundError:
        pass
    
    return model, scaler, feature_columns, categorical_options, threshold

# ==================== FUNGSI PREPROCESSING ====================
def preprocess_input(input_data, feature_columns, scaler):
    """Preprocess input data sebelum prediksi"""
    
    # Buat DataFrame dari input
    df_input = pd.DataFrame([input_data])
    
    # Kolom kategorik yang perlu di-encode
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
    
    # Pastikan semua kolom yang diperlukan ada
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Urutkan kolom sesuai dengan training
    df_encoded = df_encoded[feature_columns]
    
    # Scaling
    df_scaled = scaler.transform(df_encoded)
    
    return df_scaled

# ==================== MAIN APP ====================
def main():
    # Header dengan gradient
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Prediksi Customer Churn</h1>
        <p>Sistem Prediksi Pelanggan Berhenti Berlangganan menggunakan Artificial Neural Network (ANN)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load artifacts
    try:
        model, scaler, feature_columns, categorical_options, threshold = load_artifacts()
        st.success("✅ Model berhasil dimuat! Silakan isi data pelanggan di sidebar kiri.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.info("""
        **Pastikan file berikut ada di folder yang sama:**
        - `customer_churn_ann_model.h5` (Model ANN)
        - `scaler.pkl` (Scaler untuk normalisasi)
        - `feature_columns.json` (Daftar kolom fitur)
        - `categorical_options.json` (Opsi kategori)
        - `threshold.json` (Ambang probabilitas terbaik dari training, opsional)
        """)
        return
    
    # Info box
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        <div class="info-card">
            <h4>📋 Tentang Aplikasi</h4>
            <p>Aplikasi ini memprediksi kemungkinan pelanggan untuk berhenti berlangganan (churn) 
            berdasarkan data profil dan layanan yang digunakan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Cara Penggunaan</h4>
            <p>1. Isi semua data pelanggan di sidebar kiri<br>
            2. Klik tombol "Prediksi Sekarang"<br>
            3. Lihat hasil prediksi dan rekomendasi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR INPUT ====================
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>📝 Data Pelanggan</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.form("prediction_form"):
        # ===== INFORMASI PERSONAL =====
        st.markdown("### 👤 Informasi Personal")
        st.markdown("<p class='tooltip-text'>Data dasar mengenai pelanggan</p>", unsafe_allow_html=True)
        
        gender = st.selectbox(
            "Jenis Kelamin",
            options=['Female', 'Male'],
            format_func=lambda x: "Perempuan" if x == "Female" else "Laki-laki",
            help="Pilih jenis kelamin pelanggan"
        )
        
        senior_citizen = st.selectbox(
            "Warga Lanjut Usia (>65 tahun)?",
            options=[0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            help="Apakah pelanggan berusia di atas 65 tahun?"
        )
        
        partner = st.selectbox(
            "Memiliki Pasangan?",
            options=['Yes', 'No'],
            format_func=lambda x: "Ya" if x == "Yes" else "Tidak",
            help="Apakah pelanggan memiliki pasangan/suami/istri?"
        )
        
        dependents = st.selectbox(
            "Memiliki Tanggungan?",
            options=['Yes', 'No'],
            format_func=lambda x: "Ya" if x == "Yes" else "Tidak",
            help="Apakah pelanggan memiliki tanggungan (anak, orang tua, dll)?"
        )
        
        st.markdown("---")
        
        # ===== INFORMASI LAYANAN =====
        st.markdown("### 📡 Informasi Layanan")
        st.markdown("<p class='tooltip-text'>Detail layanan yang digunakan pelanggan</p>", unsafe_allow_html=True)
        
        tenure = st.slider(
            "Lama Berlangganan (bulan)",
            min_value=0,
            max_value=72,
            value=12,
            help="Berapa bulan pelanggan sudah berlangganan?"
        )
        
        phone_service = st.selectbox(
            "Layanan Telepon?",
            options=['Yes', 'No'],
            format_func=lambda x: "Ya" if x == "Yes" else "Tidak",
            help="Apakah pelanggan menggunakan layanan telepon?"
        )
        
        multiple_lines = st.selectbox(
            "Jalur Telepon Ganda?",
            options=['No', 'Yes', 'No phone service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No phone service': 'Tidak ada layanan telepon'
            }[x],
            help="Apakah pelanggan memiliki lebih dari satu jalur telepon?"
        )
        
        internet_service = st.selectbox(
            "Jenis Layanan Internet",
            options=['DSL', 'Fiber optic', 'No'],
            format_func=lambda x: {
                'DSL': 'DSL (Digital Subscriber Line)',
                'Fiber optic': 'Fiber Optik (Kecepatan Tinggi)',
                'No': 'Tidak Berlangganan Internet'
            }[x],
            help="Jenis koneksi internet yang digunakan"
        )
        
        online_security = st.selectbox(
            "Keamanan Online?",
            options=['No', 'Yes', 'No internet service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No internet service': 'Tidak ada layanan internet'
            }[x],
            help="Apakah pelanggan berlangganan layanan keamanan online?"
        )
        
        online_backup = st.selectbox(
            "Backup Online?",
            options=['No', 'Yes', 'No internet service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No internet service': 'Tidak ada layanan internet'
            }[x],
            help="Apakah pelanggan berlangganan layanan backup online?"
        )
        
        device_protection = st.selectbox(
            "Perlindungan Perangkat?",
            options=['No', 'Yes', 'No internet service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No internet service': 'Tidak ada layanan internet'
            }[x],
            help="Apakah pelanggan berlangganan perlindungan perangkat?"
        )
        
        tech_support = st.selectbox(
            "Dukungan Teknis?",
            options=['No', 'Yes', 'No internet service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No internet service': 'Tidak ada layanan internet'
            }[x],
            help="Apakah pelanggan berlangganan layanan dukungan teknis?"
        )
        
        streaming_tv = st.selectbox(
            "Streaming TV?",
            options=['No', 'Yes', 'No internet service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No internet service': 'Tidak ada layanan internet'
            }[x],
            help="Apakah pelanggan berlangganan layanan streaming TV?"
        )
        
        streaming_movies = st.selectbox(
            "Streaming Film?",
            options=['No', 'Yes', 'No internet service'],
            format_func=lambda x: {
                'No': 'Tidak',
                'Yes': 'Ya',
                'No internet service': 'Tidak ada layanan internet'
            }[x],
            help="Apakah pelanggan berlangganan layanan streaming film?"
        )
        
        st.markdown("---")
        
        # ===== INFORMASI KONTRAK & PEMBAYARAN =====
        st.markdown("### 💳 Kontrak & Pembayaran")
        st.markdown("<p class='tooltip-text'>Informasi kontrak dan metode pembayaran</p>", unsafe_allow_html=True)
        
        contract = st.selectbox(
            "Jenis Kontrak",
            options=['Month-to-month', 'One year', 'Two year'],
            format_func=lambda x: {
                'Month-to-month': 'Bulanan (Tanpa Kontrak)',
                'One year': 'Kontrak 1 Tahun',
                'Two year': 'Kontrak 2 Tahun'
            }[x],
            help="Jenis kontrak langganan pelanggan"
        )
        
        paperless_billing = st.selectbox(
            "Tagihan Elektronik?",
            options=['Yes', 'No'],
            format_func=lambda x: "Ya (Email/Online)" if x == "Yes" else "Tidak (Kertas/Fisik)",
            help="Apakah pelanggan menerima tagihan secara elektronik?"
        )
        
        payment_method = st.selectbox(
            "Metode Pembayaran",
            options=['Electronic check', 'Mailed check', 
                    'Bank transfer (automatic)', 'Credit card (automatic)'],
            format_func=lambda x: {
                'Electronic check': 'Cek Elektronik',
                'Mailed check': 'Cek via Pos',
                'Bank transfer (automatic)': 'Transfer Bank (Otomatis)',
                'Credit card (automatic)': 'Kartu Kredit (Otomatis)'
            }[x],
            help="Metode pembayaran yang digunakan pelanggan"
        )
        
        monthly_charges = st.number_input(
            "Tagihan Bulanan (USD $)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=0.01,
            help="Jumlah tagihan bulanan pelanggan dalam USD"
        )
        
        total_charges = st.number_input(
            "Total Tagihan Keseluruhan (USD $)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=0.01,
            help="Total tagihan sejak pertama berlangganan"
        )
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)
    
    # ==================== HASIL PREDIKSI ====================
    if submitted:
        # Kumpulkan input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Preprocess dan prediksi
        try:
            with st.spinner("🔄 Sedang memproses prediksi..."):
                X_input = preprocess_input(input_data, feature_columns, scaler)
                prediction_proba = model.predict(X_input, verbose=0)[0][0]
                prediction = 1 if prediction_proba > threshold else 0
            
            # Divider
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            
            # Header hasil
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2>📊 Hasil Prediksi</h2>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Ambang probabilitas yang digunakan: {threshold:.2f} (berdasarkan F1 terbaik)")
            
            # Metrics dalam 3 kolom
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style='color: #666; margin-bottom: 0.5rem;'>Probabilitas Churn</h4>
                    <h2 style='color: #667eea; margin: 0;'>{prediction_proba*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="metric-card metric-card-churn">
                        <h4 style='margin-bottom: 0.5rem;'>Status Prediksi</h4>
                        <h2 style='margin: 0;'>🔴 CHURN</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card metric-card-safe">
                        <h4 style='margin-bottom: 0.5rem;'>Status Prediksi</h4>
                        <h2 style='margin: 0;'>🟢 AMAN</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                confidence = max(prediction_proba, 1-prediction_proba) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style='color: #666; margin-bottom: 0.5rem;'>Tingkat Keyakinan</h4>
                    <h2 style='color: #667eea; margin: 0;'>{confidence:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Hasil detail dan rekomendasi
            if prediction == 1:
                st.markdown("""
                <div class="result-churn">
                    <h2>⚠️ PERINGATAN</h2>
                    <p style='font-size: 1.2rem;'>Pelanggan ini diprediksi akan <strong>BERHENTI BERLANGGANAN (CHURN)</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 Rekomendasi Tindakan")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("""
                    <div class="info-card" style="border-left-color: #ff6b6b;">
                        <h4>📞 Tindakan Segera</h4>
                        <ul>
                            <li>Hubungi pelanggan untuk memahami keluhan</li>
                            <li>Tawarkan konsultasi gratis</li>
                            <li>Survey kepuasan pelanggan</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with rec_col2:
                    st.markdown("""
                    <div class="info-card" style="border-left-color: #ff6b6b;">
                        <h4>🎁 Penawaran Khusus</h4>
                        <ul>
                            <li>Diskon perpanjangan kontrak</li>
                            <li>Upgrade layanan gratis sementara</li>
                            <li>Bonus kuota atau fitur tambahan</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-safe">
                    <h2>✅ PELANGGAN LOYAL</h2>
                    <p style='font-size: 1.2rem;'>Pelanggan ini diprediksi akan <strong>TETAP BERLANGGANAN</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 Rekomendasi")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("""
                    <div class="info-card" style="border-left-color: #26de81;">
                        <h4>🤝 Pertahankan Loyalitas</h4>
                        <ul>
                            <li>Pertahankan kualitas layanan</li>
                            <li>Berikan apresiasi pelanggan setia</li>
                            <li>Program loyalitas/reward</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with rec_col2:
                    st.markdown("""
                    <div class="info-card" style="border-left-color: #26de81;">
                        <h4>📈 Peluang Upselling</h4>
                        <ul>
                            <li>Tawarkan upgrade layanan</li>
                            <li>Paket bundling dengan diskon</li>
                            <li>Fitur premium baru</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualisasi Gauge
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            st.markdown("### 📈 Visualisasi Probabilitas Churn")
            
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilitas Churn (%)", 'font': {'size': 20, 'color': '#333'}},
                number={'suffix': '%', 'font': {'size': 40, 'color': '#333'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333",
                            'ticktext': ['0%', '25%', '50%', '75%', '100%'],
                            'tickvals': [0, 25, 50, 75, 100]},
                    'bar': {'color': "#667eea", 'thickness': 0.75},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#ccc",
                    'steps': [
                        {'range': [0, 25], 'color': '#26de81', 'name': 'Aman'},
                        {'range': [25, 50], 'color': '#fed330', 'name': 'Waspada'},
                        {'range': [50, 75], 'color': '#fd9644', 'name': 'Berisiko'},
                        {'range': [75, 100], 'color': '#fc5c65', 'name': 'Bahaya'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(l=30, r=30, t=60, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#333'}
            )
            
            # Legend untuk warna
            col_gauge, col_legend = st.columns([2, 1])
            
            with col_gauge:
                st.plotly_chart(fig, use_container_width=True)
            
            with col_legend:
                st.markdown("""
                <div style='padding: 1rem; background: #f8f9fa; border-radius: 10px;'>
                    <h4>📊 Keterangan Warna:</h4>
                    <p>🟢 <strong>0-25%</strong>: Aman (Kemungkinan churn rendah)</p>
                    <p>🟡 <strong>25-50%</strong>: Waspada (Perlu perhatian)</p>
                    <p>🟠 <strong>50-75%</strong>: Berisiko (Kemungkinan churn tinggi)</p>
                    <p>🔴 <strong>75-100%</strong>: Bahaya (Sangat mungkin churn)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan data input
            with st.expander("📋 Lihat Detail Data Input"):
                # Convert to Indonesian labels
                input_display = {
                    'Jenis Kelamin': "Perempuan" if gender == "Female" else "Laki-laki",
                    'Warga Lanjut Usia': "Ya" if senior_citizen == 1 else "Tidak",
                    'Memiliki Pasangan': "Ya" if partner == "Yes" else "Tidak",
                    'Memiliki Tanggungan': "Ya" if dependents == "Yes" else "Tidak",
                    'Lama Berlangganan': f"{tenure} bulan",
                    'Layanan Telepon': "Ya" if phone_service == "Yes" else "Tidak",
                    'Jalur Telepon Ganda': {'No': 'Tidak', 'Yes': 'Ya', 'No phone service': 'Tidak ada layanan telepon'}[multiple_lines],
                    'Layanan Internet': {'DSL': 'DSL', 'Fiber optic': 'Fiber Optik', 'No': 'Tidak Ada'}[internet_service],
                    'Keamanan Online': {'No': 'Tidak', 'Yes': 'Ya', 'No internet service': 'Tidak ada internet'}[online_security],
                    'Backup Online': {'No': 'Tidak', 'Yes': 'Ya', 'No internet service': 'Tidak ada internet'}[online_backup],
                    'Perlindungan Perangkat': {'No': 'Tidak', 'Yes': 'Ya', 'No internet service': 'Tidak ada internet'}[device_protection],
                    'Dukungan Teknis': {'No': 'Tidak', 'Yes': 'Ya', 'No internet service': 'Tidak ada internet'}[tech_support],
                    'Streaming TV': {'No': 'Tidak', 'Yes': 'Ya', 'No internet service': 'Tidak ada internet'}[streaming_tv],
                    'Streaming Film': {'No': 'Tidak', 'Yes': 'Ya', 'No internet service': 'Tidak ada internet'}[streaming_movies],
                    'Jenis Kontrak': {'Month-to-month': 'Bulanan', 'One year': '1 Tahun', 'Two year': '2 Tahun'}[contract],
                    'Tagihan Elektronik': "Ya" if paperless_billing == "Yes" else "Tidak",
                    'Metode Pembayaran': {'Electronic check': 'Cek Elektronik', 'Mailed check': 'Cek via Pos', 'Bank transfer (automatic)': 'Transfer Bank', 'Credit card (automatic)': 'Kartu Kredit'}[payment_method],
                    'Tagihan Bulanan': f"${monthly_charges:.2f}",
                    'Total Tagihan': f"${total_charges:.2f}"
                }
                
                # Display as 2 columns
                col_data1, col_data2 = st.columns(2)
                items = list(input_display.items())
                mid = len(items) // 2
                
                with col_data1:
                    for key, value in items[:mid]:
                        st.markdown(f"**{key}:** {value}")
                
                with col_data2:
                    for key, value in items[mid:]:
                        st.markdown(f"**{key}:** {value}")
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
            st.exception(e)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎓 <strong>Tugas Akhir 11 - Praktikum Machine Learning</strong></p>
        <p>Artificial Neural Network (ANN) - Prediksi Customer Churn</p>
        <p style='font-size: 0.8rem; color: #999;'>Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
