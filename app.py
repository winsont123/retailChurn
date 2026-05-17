import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 1. Konfigurasi Halaman Utama
st.set_page_config(page_title="Retail Churn AI Portal", page_icon="🛍️", layout="wide")

# 2. Fungsi Memuat Model Secara Dinamis
@st.cache_resource
def load_selected_model(window_choice, optimization_choice):
    if window_choice == "3 Months (Scenario A)":
        return joblib.load('xgb_3m_base.pkl'), "Model standar berbasis tren jangka pendek (3 bulan)."
    elif window_choice == "6 Months (Scenario B)":
        return joblib.load('xgb_6m_base.pkl'), "Model area transisi (6 bulan). Rentan terhadap noise akibat perubahan perilaku."
    else: # 12 Months
        if optimization_choice == "Hyper-Tuned (Aggressive Churn Recall)":
            return joblib.load('xgb_12m_tuned.pkl'), "Model optimasi tertinggi. Sangat sensitif mendeteksi pelanggan yang berpotensi kabur."
        else:
            return joblib.load('xgb_12m_base.pkl'), "Model dasar dengan performa seimbang pada siklus tahunan penuh."

# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.title("⚙️ AI Control Panel")
    st.markdown("Sesuaikan konfigurasi data historis dan kecerdasan model secara real-time.")
    st.markdown("---")
    
    # Filter 1: Memilih Jendela Waktu (Sliding Window Concept)
    window_selection = st.selectbox(
        "1. Durasi Jendela Observasi:",
        ["12 Months (Scenario C - Winner)", "3 Months (Scenario A)", "6 Months (Scenario B)"]
    )
    
    # Filter 2: Memilih Optimasi (Hanya muncul jika memilih 12 bulan)
    if "12 Months" in window_selection:
        opt_selection = st.radio(
            "2. Tingkat Optimasi Model:",
            ["Hyper-Tuned (Aggressive Churn Recall)", "Base Model (Balanced)"]
        )
    else:
        opt_selection = "Base Model (Balanced)"
        st.info("💡 Hyper-tuning diterapkan khusus pada Skenario C sebagai pemenang evaluasi.")

    # Memuat model aktif berdasarkan pilihan user
    active_model, model_desc = load_selected_model(window_selection, opt_selection)
    
    st.markdown("---")
    st.subheader("🤖 Info Model Aktif:")
    st.caption(model_desc)
    if "Hyper-Tuned" in opt_selection:
        st.warning("⚠️ Mode Agresif Aktif: Model diprioritaskan untuk mengamankan retensi kelas minoritas.")

# ==========================================
# TAMPILAN UTAMA DASHBOARD
# ==========================================
st.title("🛍️ AI Customer Retention Dashboard")
st.markdown("Sistem Pendukung Keputusan Berbasis Komparasi Jendela Waktu untuk Mitigasi *Concept Drift* pada Sektor E-Commerce.")
st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Per-Customer Prediction", "📊 Batch Processing & Enterprise Analytics"])

# ==========================================
# TAB 1: PREDIKSI INDIVIDU
# ==========================================
with tab1:
    st.markdown("### 🔍 Input Atribut Perilaku Pelanggan (RFMT)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        recency = st.number_input("Recency (Hari sejak transaksi terakhir)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (Total invoice yang ada sekarang)", min_value=1, value=5)
    with col3:
        monetary = st.number_input("Monetary (Total nilai belanja dalam USD)", min_value=0.0, value=500.0)
    with col4:
        time_val = st.number_input("Time (Rata-rata jeda hari antar belanja)", min_value=0.0, value=10.0)

    if st.button("🚀 Jalankan Analisis Prediksi", type="primary", use_container_width=True):
        input_df = pd.DataFrame([[recency, frequency, monetary, time_val]],
                                 columns=['Recency', 'Frequency', 'Monetary', 'Time'])
        
        prob_churn = active_model.predict_proba(input_df)[0][1]
        risk_percentage = prob_churn * 100

        st.markdown("---")
        col_gauge, col_recom = st.columns([1, 1.5])
        
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                number={'suffix': "%", 'font': {'size': 50}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "⚠️ Skor Risiko Churn", 'font': {'size': 22}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "#00cc96"}, 
                        {'range': [40, 70], 'color': "#FFA15A"}, 
                        {'range': [70, 100], 'color': "#EF553B"}],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_percentage}
                }
            ))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Penjelasan Tingkat Kepedean AI (Interpretability)
            if risk_percentage >= 80:
                st.error("🧠 **AI Confidence:** Sangat Yakin (Pola perilaku menunjukkan kecenderungan kuat meninggalkan platform).")
            elif risk_percentage >= 50:
                st.warning("🧠 **AI Confidence:** Cukup Yakin (Menunjukkan tanda-tanda awal penurunan loyalitas).")
            elif risk_percentage >= 20:
                st.info("🧠 **AI Confidence:** Cukup Yakin (Aktivitas belanja masih dalam batas wajar dan stabil).")
            else:
                st.success("🧠 **AI Confidence:** Sangat Yakin (Profil menunjukkan karakteristik pelanggan setia).")

        with col_recom:
            st.markdown("### 💡 AI Business Strategy Recommendation")
            if risk_percentage > 50:
                st.error("**STATUS: HIGH RISK (PROSFEK CHURN)**")
                st.markdown("#### 🛠️ Tindakan Retensi Direkomendasikan:")
                if "Hyper-Tuned" in opt_selection:
                    st.caption("ℹ️ *Rekomendasi disesuaikan dengan sensitivitas model Hyper-Tuned yang agresif.*")
                
                if frequency <= 2:
                    st.info("- **Analisis Masalah:** Pelanggan baru dengan retensi rendah.\n- **Solusi Eksekutif:** Picu kembali minat beli dengan meluncurkan kampanye email penawaran produk pelengkap disertai voucher diskon 25%.")
                elif monetary >= 1000:
                    st.warning("- **Analisis Masalah:** Risiko kehilangan aset berharga (VIP Customer)!\n- **Solusi Eksekutif:** Segera delegasikan tim hubungan pelanggan untuk memberikan apresiasi berupa poin ganda atau cinderamata eksklusif.")
                else:
                    st.info("- **Solusi Eksekutif:** Masukkan nomor ID pelanggan ke dalam target iklan digital harian (*retargeting campaign*).")
            else:
                st.success("**STATUS: SAFE (PELANGGAN LOYAL)**")
                st.markdown("#### 🛠️ Tindakan Pemeliharaan:")
                st.info("- Pertahankan standar layanan untuk mempertahankan loyalitas jangka panjang.\n- Berikan kode *referral* khusus agar mereka dapat membantu mendatangkan pelanggan baru secara organik.")

# ==========================================
# TAB 2: PROSES MASSAL (RAW DATA)
# ==========================================
with tab2:
    st.markdown("### 📊 Proses Analisis Data Transaksi Mentah")
    st.write("Sistem otomatis melakukan pemotongan waktu (time-windowing) sesuai skenario model yang Anda pilih di Sidebar, mengekstrak RFMT, dan menghitung risiko churn.")
    
    uploaded_file = st.file_uploader("Upload File Laporan Transaksi (.CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            with st.spinner(f"Memotong data transaksi dan menjalankan {window_selection}..."):
                df_raw = pd.read_csv(uploaded_file)
                df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
                df_raw['Total_Price'] = df_raw['Quantity'] * df_raw['Price']
                
               # 1. Tentukan Titik Acuan Waktu (Snapshot Date) dan Tanggal Paling Awal
                snapshot_date = df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)
                min_date_in_data = df_raw['InvoiceDate'].min()
                
                # Hitung rentang waktu data aktual (dalam hari)
                data_duration_days = (snapshot_date - min_date_in_data).days
                
                # 2. LOGIKA SLIDING WINDOW (Pemotongan Data Otomatis!)
                if "3 Months" in window_selection:
                    required_days = 90
                    start_date = snapshot_date - pd.DateOffset(months=3)
                elif "6 Months" in window_selection:
                    required_days = 180
                    start_date = snapshot_date - pd.DateOffset(months=6)
                else: # 12 Months
                    required_days = 365
                    start_date = snapshot_date - pd.DateOffset(months=12)
                
                # --- SISTEM PERINGATAN JIKA DATA WAKTU TIDAK PAS ---
                if data_duration_days < required_days:
                    st.warning(f"⚠️ **Peringatan Skala Data:** Model yang Anda pilih membutuhkan data historis selama **{required_days} hari**, namun data yang diunggah hanya berdurasi **{data_duration_days} hari**. Hasil prediksi probabilitas mungkin menjadi pesimistis (bias terhadap Churn) karena fitur Frequency dan Monetary belum terakumulasi penuh.")
                
                # Buang data yang lebih lama dari start_date
                df_filtered = df_raw[df_raw['InvoiceDate'] >= start_date].copy()
                
                if df_filtered.empty:
                    st.error("Data CSV kosong setelah dipotong sesuai jendela waktu. Pastikan rentang waktu data Anda mencukupi.")
                else:
                    # 3. Hitung RFMT MENGGUNAKAN DATA YANG SUDAH DIPOTONG
                    rfm = df_filtered.groupby('Customer ID').agg({
                        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                        'Invoice': 'nunique',
                        'Total_Price': 'sum'
                    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Total_Price': 'Monetary'})
                    
                    def calc_time(dates):
                        dates = dates.sort_values()
                        if len(dates) > 1:
                            return dates.diff().mean().days
                        return 0.0
                    
                    t_df = df_filtered.groupby('Customer ID')['InvoiceDate'].apply(calc_time).reset_index(name='Time')
                    rfmt_df = pd.merge(rfm, t_df, on='Customer ID')
                    
                    # 4. Prediksi Model
                    X_batch = rfmt_df[['Recency', 'Frequency', 'Monetary', 'Time']]
                    rfmt_df['Churn_Probability'] = active_model.predict_proba(X_batch)[:, 1]
                    rfmt_df['Status'] = ["Churn Risk" if p > 0.5 else "Loyal" for p in rfmt_df['Churn_Probability']]
                    
                    st.success(f"✅ Pemrosesan Selesai! Data diekstrak dari periode: {start_date.date()} hingga {snapshot_date.date()}")
                    
                    # EXECUTIVE SUMMARY METRICS
                    st.markdown("### 📈 Executive Analytical Summary")
                    total_cust = len(rfmt_df)
                    churn_cust = len(rfmt_df[rfmt_df['Status'] == 'Churn Risk'])
                    revenue_at_risk = rfmt_df[rfmt_df['Status'] == 'Churn Risk']['Monetary'].sum()
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric(label="👥 Pelanggan Aktif di Periode Ini", value=total_cust)
                    kpi2.metric(label="⚠️ Teridentifikasi Risiko Churn", value=f"{churn_cust} Orang", delta=f"{(churn_cust/total_cust)*100:.1f}% Risiko", delta_color="inverse")
                    kpi3.metric(label="💸 Potensi Nilai Terancam", value=f"${revenue_at_risk:,.2f}")
                    
                    # GRAPH VISUALIZATIONS
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        fig_p = px.pie(rfmt_df, names='Status', title="Proporsi Status Kelas Pelanggan", 
                                       color='Status', color_discrete_map={'Loyal':'#00cc96', 'Churn Risk':'#EF553B'}, hole=0.3)
                        st.plotly_chart(fig_p, use_container_width=True)
                    with col_c2:
                        fig_h = px.histogram(rfmt_df, x='Status', y='Monetary', title="Volume Kontribusi Uang Berdasarkan Status",
                                             color='Status', color_discrete_map={'Loyal':'#00cc96', 'Churn Risk':'#EF553B'}, histfunc='sum')
                        st.plotly_chart(fig_h, use_container_width=True)

                    st.markdown("### 📋 Tabel Deteksi Nilai Tambah Pelanggan")
                    st.dataframe(rfmt_df.style.format({'Monetary': '${:.2f}', 'Churn_Probability': '{:.1%}', 'Time': '{:.1f}'}))
                    
                    # DOWNLOAD REPORT
                    csv_data = rfmt_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Unduh Dokumen Hasil Prediksi (CSV)",
                        data=csv_data,
                        file_name=f"Churn_Report_{window_selection}.csv",
                        mime="text/csv",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Gagal memproses berkas data transaksi. Detail: {e}")