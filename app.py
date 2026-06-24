import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Retail Churn AI Portal", page_icon="🛍️", layout="wide")

# ==========================================
# 2. FUNGSI MEMUAT MODEL (CACHE)
# ==========================================
@st.cache_resource
def load_selected_model(window_choice):
    try:
        if "3 Months" in window_choice:
            return joblib.load('model_3_months.pkl'), "Model jangka pendek (3 bulan). Sensitif terhadap fluktuasi sesaat, efektif sebagai sinyal peringatan dini."
        elif "6 Months" in window_choice:
            return joblib.load('model_6_months.pkl'), "Model menengah (6 bulan). Menangkap transisi tren paruh tahunan."
        else: # 12 Months
            return joblib.load('model_12_months.pkl'), "Model historis penuh (12 bulan / Winner). Paling stabil, memiliki pemahaman akurat terhadap anomali siklus hari raya."
    except Exception as e:
        st.error(f"Gagal memuat model {window_choice}. Pastikan file .pkl berada di direktori aktif. Error: {e}")
        return None, "Sistem tidak siap."

# ==========================================
# 3. SIDEBAR (CONTROL PANEL)
# ==========================================
with st.sidebar:
    st.title("AI Control Panel")
    st.markdown("Sesuaikan lensa observasi algoritma secara *real-time*.")
    st.markdown("---")
    
    window_selection = st.selectbox(
        "Pilih Jendela Waktu Observasi:",
        ["12 Months (Scenario C - Winner)", "6 Months (Scenario B)", "3 Months (Scenario A)"],
        index=0,
        help="Menentukan seberapa jauh ke belakang AI harus menganalisis riwayat belanja pelanggan."
    )
    
    active_model, model_desc = load_selected_model(window_selection)
    
    st.markdown("---")
    st.subheader("Info Model Aktif:")
    st.info(model_desc)
    
    with st.expander("📖 Panduan Membaca RFMT (Klik di sini)"):
        st.markdown("""
        * **Recency:** Jumlah hari sejak belanja terakhir. Makin kecil = Makin baik.
        * **Frequency:** Total transaksi/invoice sukses. Makin besar = Makin loyal.
        * **Monetary:** Akumulasi uang yang dihabiskan. Makin besar = Makin VIP.
        * **Time:** Rata-rata jeda hari antar belanja. Menunjukkan rutinitas.
        """)

# ==========================================
# 4. TAMPILAN UTAMA DASHBOARD
# ==========================================

st.title("Retail Customer Retention Dashboard")
st.markdown("Sistem Analitik berbasis *Machine Learning* untuk Memitigasi Risiko *Customer Churn*.")
st.markdown("---")

tab1, tab2 = st.tabs(["Per-Customer Prediction", "Batch Processing & Enterprise Analytics"])

# ==========================================
# TAB 1: PREDIKSI INDIVIDU (PER-CUSTOMER)
# ==========================================
with tab1:
    st.markdown("### 🔍 Input Profil Perilaku Pelanggan (RFMT)")
    
    if active_model:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            recency = st.number_input("Recency (Hari sejak belanja terakhir)", min_value=0, value=30)
        with col2:
            frequency = st.number_input("Frequency (Total keranjang belanja)", min_value=1, value=5)
        with col3:
            monetary = st.number_input("Monetary (Total belanja dalam USD $)", min_value=0.0, value=250.00, step=50.0)
        with col4:
            time_val = st.number_input("Time (Rata-rata jeda hari antar belanja)", min_value=0.0, value=14.0, step=1.0)

        if st.button("Jalankan Analisis AI", type="primary", use_container_width=True):
            input_df = pd.DataFrame([[recency, frequency, monetary, time_val]],
                                     columns=['Recency', 'Frequency', 'Monetary', 'Time'])
            
            prob_churn = active_model.predict_proba(input_df)[0][1]
            risk_percentage = prob_churn * 100

            st.markdown("---")
            col_gauge, col_recom = st.columns([1.2, 1.8])
            
            with col_gauge:
                # GAUGE CHART (Transparan & Elegan)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percentage,
                    number={'suffix': "%", 'font': {'size': 38, 'color': '#31333F'}},
                    title={'text': "⚠️ Skor Risiko Churn", 'font': {'size': 18, 'color': '#31333F'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': "rgba(0,0,0,0.8)", 'thickness': 0.3},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': "#00cc96"},   # Hijau
                            {'range': [40, 70], 'color': "#FFA15A"},  # Oranye
                            {'range': [70, 100], 'color': "#EF553B"}  # Merah
                        ],
                        'threshold': {'line': {'color': "#31333F", 'width': 5}, 'thickness': 0.8, 'value': risk_percentage}
                    }
                ))
                fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True)
                
            with col_recom:
                st.markdown("### 💡 AI Business Strategy Recommendation")
                if risk_percentage >= 70:
                    st.error("**STATUS: HIGH RISK (PROSPEK CHURN TINGGI)**")
                    st.write("**Diagnosis:** Pelanggan menunjukkan gejala detasemen permanen dari platform.")
                    st.write("**Tindakan Retensi:** Segera lakukan intervensi via *Customer Success Call* atau berikan *Voucher Cashback* 30% tanpa minimum pembelian.")
                elif risk_percentage >= 40:
                    st.warning("**STATUS: MEDIUM RISK (FASE MERENGGANG)**")
                    st.write("**Diagnosis:** Rutinitas interval belanja mulai melambat.")
                    st.write("**Tindakan Retensi:** Masukkan ID pelanggan ke dalam kampanye iklan penargetan ulang (*Automated Email Retargeting*).")
                else:
                    st.success("**STATUS: SAFE (PELANGGAN LOYAL)**")
                    st.write("**Diagnosis:** Karakteristik RFMT sangat sehat, didominasi frekuensi kedatangan yang rapat.")
                    st.write("**Tindakan Retensi:** Jaga kualitas SLA pengiriman. Tawarkan pendaftaran keanggotaan *Tier VIP*.")

# ==========================================
# TAB 2: PROSES MASSAL (BATCH PROCESSING)
# ==========================================
with tab2:
    st.markdown("### Executive Batch Processing & Enterprise Analytics")
    st.info("Unggah berkas riwayat transaksi CSV. Sistem akan memotong rentang waktu secara otomatis, mengekstrak fitur RFMT, dan menyajikan dasbor performa retensi eksekutif.")
    
    with st.expander("Lihat Spesifikasi Format CSV"):
        st.write("Sistem mengharuskan berkas CSV memiliki 4 header persis seperti ini:")
        st.code("Customer ID, InvoiceDate, Quantity, Price", language="text")
    
    uploaded_file = st.file_uploader("Unggah Laporan Transaksi (.CSV)", type=["csv"])
    
    if uploaded_file is not None and active_model is not None:
        try:
            with st.spinner("Mengeksekusi ETL Pipeline & Inferensi XGBoost... (Akan memakan waktu 1-2 menit)"):
                df_raw = pd.read_csv(uploaded_file)
                
                # Standarisasi kolom
                if 'InvoiceDate' not in df_raw.columns or 'Customer ID' not in df_raw.columns:
                    st.error("Kegagalan Pipa Data: Kolom 'Customer ID' atau 'InvoiceDate' tidak ditemukan.")
                else:
                    df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
                    df_raw['Total_Price'] = df_raw['Quantity'] * df_raw['Price']
                    
                    snapshot_date = df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)
                    
                    # Logika Jendela Mundur
                    if "3 Months" in window_selection:
                        start_date = snapshot_date - pd.DateOffset(months=3)
                    elif "6 Months" in window_selection:
                        start_date = snapshot_date - pd.DateOffset(months=6)
                    else: 
                        start_date = snapshot_date - pd.DateOffset(months=12)

                    df_filtered = df_raw[df_raw['InvoiceDate'] >= start_date].copy()
                    
                    if df_filtered.empty:
                        st.error("Populasi data kosong pada rentang observasi yang dipilih.")
                    else:
                        # EKSTRAKSI RFM
                        rfm = df_filtered.groupby('Customer ID').agg({
                            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                            'Invoice': 'nunique',
                            'Total_Price': 'sum'
                        }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Total_Price': 'Monetary'})
                        
                        # EKSTRAKSI TIME (T) - Menggunakan fungsi aman berpresisi float
                        def get_time_mean(series):
                            dates = series.sort_values().unique()
                            if len(dates) > 1:
                                return float(np.mean(np.diff(dates)) / np.timedelta64(1, 'D'))
                            return 0.0
                        
                        t_df = df_filtered.groupby('Customer ID')['InvoiceDate'].apply(get_time_mean).reset_index(name='Time')
                        rfmt_df = pd.merge(rfm.reset_index(), t_df, on='Customer ID')
                        
                        # SCORING MODEL
                        X_batch = rfmt_df[['Recency', 'Frequency', 'Monetary', 'Time']]
                        rfmt_df['Churn_Probability'] = active_model.predict_proba(X_batch)[:, 1]
                        rfmt_df['Status'] = ["Churn Risk" if p > 0.5 else "Loyal" for p in rfmt_df['Churn_Probability']]
                        
                        # ==========================================
                        # RENDERING KPI DASHBOARD (Gaya PPT Slide 16)
                        # ==========================================
                        st.markdown("### Executive Analytical Summary")
                        
                        total_cust = len(rfmt_df)
                        churn_cust = len(rfmt_df[rfmt_df['Status'] == 'Churn Risk'])
                        revenue_at_risk = rfmt_df[rfmt_df['Status'] == 'Churn Risk']['Monetary'].sum()
                        
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("Pelanggan Aktif di Periode Ini", f"{total_cust:,}")
                        kpi2.metric("Teridentifikasi Risiko Churn", f"{churn_cust:,} Orang", f"{(churn_cust/total_cust)*100:.1f}% Risiko", delta_color="inverse")
                        kpi3.metric("Potensi Nilai Terancam", f"${revenue_at_risk:,.2f}")
                        
                        # GRAFIK KOMPARASI (KIRI PIE, KANAN BAR)
                        col_g1, col_g2 = st.columns(2)
                        
                        with col_g1:
                            fig_pie = px.pie(rfmt_df, names='Status', title="Proporsi Status Kelas Pelanggan",
                                             color='Status', color_discrete_map={'Loyal':'#00cc96', 'Churn Risk':'#EF553B'}, hole=0.45)
                            fig_pie.update_traces(textposition='inside', textinfo='label+percent', insidetextfont=dict(size=14, color="white"))
                            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10, l=10, r=10))
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col_g2:
                            # Menggunakan bar chart murni hasil agregasi agar sumbu X rapi
                            bar_agg = rfmt_df.groupby('Status')['Monetary'].sum().reset_index()
                            fig_bar = px.bar(bar_agg, x='Status', y='Monetary', title="Volume Kontribusi Uang Berdasarkan Status",
                                             color='Status', color_discrete_map={'Loyal':'#00cc96', 'Churn Risk':'#EF553B'})
                            fig_bar.update_layout(yaxis=dict(title="sum of Monetary", tickformat="$.2s"), xaxis=dict(title="Status"),
                                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # TABEL HASIL & DOWNLOAD
                        st.subheader("Tabel Deteksi Nilai Tambah Pelanggan")
                        st.dataframe(rfmt_df.style.format({'Monetary': '${:,.2f}', 'Churn_Probability': '{:.1%}', 'Time': '{:.1f}'}), use_container_width=True)
                        
                        csv_export = rfmt_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="⬇️ Unduh Dokumen Hasil Prediksi (.CSV)", data=csv_export, file_name=f"Executive_Churn_Report_{window_selection[:2]}M.csv", mime="text/csv", type="primary")

        except Exception as e:
            st.error(f"Terjadi kegagalan pemrosesan berkas. Detail teknis: {e}")
