import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Retail Churn AI Portal", page_icon="logo.png", layout="wide")

# ==========================================
# 2. FUNGSI MEMUAT MODEL (CACHE)
# ==========================================
@st.cache_resource
def load_selected_model(window_choice):
    try:
        if window_choice == "3 Months (Scenario A)":
            return joblib.load('model_3_months.pkl'), "Model jangka pendek (3 bulan). Sensitif terhadap perubahan mendadak, tapi rentan memberikan alarm palsu."
        elif window_choice == "6 Months (Scenario B)":
            return joblib.load('model_6_months.pkl'), "Model menengah (6 bulan). Menangkap tren semi-tahunan."
        else: # 12 Months
            return joblib.load('model_12_months.pkl'), "Model jangka panjang (12 bulan). Paling stabil, akurat, dan memahami siklus belanja tahunan pelanggan."
    except FileNotFoundError:
        st.error(f"File model untuk {window_choice} tidak ditemukan! Pastikan file .pkl ada di folder yang sama.")
        return None, "Model tidak tersedia."

# ==========================================
# 3. SIDEBAR (CONTROL PANEL)
# ==========================================
with st.sidebar:
    st.title("⚙️ AI Control Panel")
    st.markdown("Sesuaikan sudut pandang analisis algoritma secara *real-time*.")
    st.markdown("---")
    
    window_selection = st.selectbox(
        "Pilih Lensa Waktu Observasi:",
        ["12 Months (Scenario C - Winner)", "6 Months (Scenario B)", "3 Months (Scenario A)"],
        help="Semakin panjang waktu observasi, AI semakin memahami pola loyalitas jangka panjang pelanggan."
    )
    
    active_model, model_desc = load_selected_model(window_selection)
    
    st.markdown("---")
    st.subheader("Info Model Aktif:")
    st.info(model_desc)
    
    # Edukasi untuk User
    with st.expander("📖 Panduan Membaca RFMT (Klik di sini)"):
        st.markdown("""
        * **Recency:** Jumlah hari sejak belanja terakhir. Makin kecil = Makin baik.
        * **Frequency:** Berapa kali belanja. Makin besar = Makin loyal.
        * **Monetary:** Total uang yang dihabiskan. Makin besar = Makin VIP.
        * **Time:** Jarak hari antar transaksi. Menunjukkan rutinitas pelanggan.
        """)

# ==========================================
# 4. TAMPILAN UTAMA DASHBOARD
# ==========================================

st.title("Retail Customer Retention Dashboard")
st.markdown("Sistem Pendukung Keputusan Cerdas untuk Mencegah Kehilangan Pelanggan (*Customer Churn*).")
st.markdown("---")

tab1, tab2 = st.tabs(["Per-Customer Prediction", "Batch Processing (Data Massal)"])

# ==========================================
# TAB 1: PREDIKSI INDIVIDU
# ==========================================
with tab1:
    st.markdown("### 🔍 Input Profil Perilaku Pelanggan")
    
    if active_model:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            recency = st.number_input("Recency (Hari sejak belanja terakhir)", min_value=0, value=30, help="Contoh: 30 artinya sudah 1 bulan tidak belanja.")
        with col2:
            frequency = st.number_input("Frequency (Total transaksi dilakukan)", min_value=1, value=5, help="Total invoice atau keranjang belanja yang pernah dibayar.")
        with col3:
            monetary = st.number_input("Monetary (Total belanja dalam Rupiah)", min_value=0.0, value=500000.0, step=50000.0)
        with col4:
            time_val = st.number_input("Time (Rata-rata jarak hari antar belanja)", min_value=0.0, value=14.0, help="Contoh: 14 artinya pelanggan ini rutin belanja setiap 2 minggu sekali.")

        if st.button("Jalankan Analisis AI", type="primary", use_container_width=True):
            # Format ke DataFrame
            input_df = pd.DataFrame([[recency, frequency, monetary, time_val]],
                                     columns=['Recency', 'Frequency', 'Monetary', 'Time'])
            
            # Prediksi Probabilitas
            prob_churn = active_model.predict_proba(input_df)[0][1]
            risk_percentage = prob_churn * 100

            st.markdown("---")
            col_gauge, col_recom = st.columns([1, 1.5])
            
            with col_gauge:
                # Visualisasi Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percentage,
                    number={'suffix': "%", 'font': {'size': 40}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "⚠️ Skor Risiko Churn", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "#00cc96"}, # Hijau (Aman)
                            {'range': [40, 70], 'color': "#FFA15A"}, # Oranye (Waspada)
                            {'range': [70, 100], 'color': "#EF553B"} # Merah (Bahaya)
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_percentage}
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            with col_recom:
                st.markdown("### 💡 AI Business Strategy Recommendation")
                if risk_percentage >= 70:
                    st.error("**STATUS: HIGH RISK (PROSPEK CHURN TINGGI)**")
                    st.markdown("#### Tindakan Retensi Direkomendasikan:")
                    if frequency <= 2:
                        st.write("- **Analisis:** Pelanggan baru yang gagal menjadi pelanggan tetap.")
                        st.write("- **Tindakan:** Kirim email penawaran produk pelengkap (Cross-selling) dengan diskon khusus 25%.")
                    else:
                        st.write("- **Analisis:** Pelanggan lama yang mengubah rutinitasnya secara drastis!")
                        st.write("- **Tindakan:** Segera hubungi via telepon (Customer Success) untuk menanyakan kendala atau berikan Voucher VIP.")
                elif risk_percentage >= 40:
                    st.warning("**STATUS: MEDIUM RISK (MULAI MENJAUH)**")
                    st.write("- **Analisis:** Pelanggan menunjukkan tanda awal penurunan aktivitas.")
                    st.write("- **Tindakan:** Masukkan ID pelanggan ini ke dalam target iklan digital (*Retargeting Campaign*).")
                else:
                    st.success("**STATUS: SAFE (PELANGGAN LOYAL)**")
                    st.write("- **Analisis:** Profil transaksi pelanggan sangat sehat dan stabil.")
                    st.write("- **Tindakan:** Pertahankan layanan. Tawarkan program *referral* agar mereka membawa teman baru.")

# ==========================================
# TAB 2: PROSES MASSAL (RAW DATA)
# ==========================================
with tab2:
    st.markdown("### Proses Analisis Data Transaksi Massal")
    st.info("Unggah file CSV berisi data historis transaksi. Sistem akan otomatis mengekstrak RFMT dan memprediksi massal berdasarkan lensa waktu yang dipilih di Sidebar.")
    
    with st.expander("Lihat Format CSV yang Dibutuhkan"):
        st.write("Pastikan file CSV Anda memiliki kolom persis seperti ini:")
        st.code("Customer ID, InvoiceDate, Quantity, Price", language="text")
    
    uploaded_file = st.file_uploader("Upload File Transaksi (.CSV)", type=["csv"])
    
    if uploaded_file is not None and active_model is not None:
        try:
            with st.spinner("Memproses data besar dan menjalankan Machine Learning..."):
                df_raw = pd.read_csv(uploaded_file)
                
                # Standarisasi kolom (Asumsi nama kolom standar)
                if 'InvoiceDate' not in df_raw.columns or 'Customer ID' not in df_raw.columns:
                    st.error("Format CSV salah! Pastikan ada kolom 'Customer ID' dan 'InvoiceDate'.")
                else:
                    df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
                    df_raw['Total_Price'] = df_raw['Quantity'] * df_raw['Price']
                    
                    # Menghitung durasi total data yang diunggah
                    snapshot_date = df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)
                    min_date_in_data = df_raw['InvoiceDate'].min()
                    data_duration_days = (snapshot_date - min_date_in_data).days

                    # Logika Pemotongan Waktu
                    if "3 Months" in window_selection:
                        required_days = 90
                        start_date = snapshot_date - pd.DateOffset(months=3)
                    elif "6 Months" in window_selection:
                        required_days = 180
                        start_date = snapshot_date - pd.DateOffset(months=6)
                    else: 
                        required_days = 365
                        start_date = snapshot_date - pd.DateOffset(months=12)
                    
                    # Menampilkan peringatan jika durasi data CSV lebih pendek dari kebutuhan model
                    if data_duration_days < required_days:
                        st.warning(f"⚠️ Peringatan Skala Data: Model yang Anda pilih idealnya membutuhkan data historis selama {required_days} hari. Namun data CSV yang diunggah hanya memiliki rentang {data_duration_days} hari. Hasil prediksi probabilitas churn mungkin menjadi bias karena riwayat transaksi pelanggan belum terakumulasi secara penuh.")

                    df_filtered = df_raw[df_raw['InvoiceDate'] >= start_date].copy()
                    
                    if df_filtered.empty:
                        st.error("Data CSV kosong di rentang waktu tersebut.")
                    else:
                        # Ekstrak RFM
                        rfm = df_filtered.groupby('Customer ID').agg({
                            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                            'Invoice': 'nunique', # Asumsi nama kolom invoice
                            'Total_Price': 'sum'
                        }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Total_Price': 'Monetary'})
                        
                        # Ekstrak Time (T)
                        def calc_time(dates):
                            dates = dates.sort_values().unique()
                            if len(dates) > 1:
                                return pd.Series(dates).diff().mean().days
                            return 0.0
                        
                        t_df = df_filtered.groupby('Customer ID')['InvoiceDate'].apply(calc_time).reset_index(name='Time')
                        rfmt_df = pd.merge(rfm, t_df, on='Customer ID')
                        
                        # PREDIKSI MASSAL
                        X_batch = rfmt_df[['Recency', 'Frequency', 'Monetary', 'Time']]
                        rfmt_df['Churn_Probability'] = active_model.predict_proba(X_batch)[:, 1]
                        rfmt_df['Status'] = ["High Risk" if p > 0.5 else "Loyal" for p in rfmt_df['Churn_Probability']]
                        
                        st.success("Pemrosesan Selesai!")
                        
                        # KPI Dashboard
                        st.markdown("### Executive Analytical Summary")
                        total_cust = len(rfmt_df)
                        churn_cust = len(rfmt_df[rfmt_df['Status'] == 'High Risk'])
                        revenue_at_risk = rfmt_df[rfmt_df['Status'] == 'High Risk']['Monetary'].sum()
                        
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("Pelanggan Aktif", total_cust)
                        kpi2.metric("Risiko Churn", f"{churn_cust} Orang", f"{(churn_cust/total_cust)*100:.1f}%", delta_color="inverse")
                        kpi3.metric("Potensi Pendapatan Hilang", f"Rp {revenue_at_risk:,.0f}")
                        
                        # Visualisasi Massal
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            fig_p = px.pie(rfmt_df, names='Status', title="Proporsi Status Pelanggan", 
                                           color='Status', color_discrete_map={'Loyal':'#00cc96', 'High Risk':'#EF553B'}, hole=0.4)
                            fig_p.update_traces(
                                textposition='inside',      
                                textinfo='label+percent',  
                                insidetextfont=dict(
                                family="Arial", 
                                size=16,              
                                color="white"           
                                )
                            )
                            st.plotly_chart(fig_p, use_container_width=True)
                        with col_c2:
                            fig_h = px.histogram(rfmt_df, x='Status', y='Monetary', title="Volume Belanja Berdasarkan Status",
                                                 color='Status', color_discrete_map={'Loyal':'#00cc96', 'High Risk':'#EF553B'}, histfunc='sum')
                            st.plotly_chart(fig_h, use_container_width=True)

                        # Tabel dan Download
                        st.dataframe(rfmt_df.style.format({'Monetary': 'Rp {:.0f}', 'Churn_Probability': '{:.1%}', 'Time': '{:.1f}'}), use_container_width=True)
                        
                        csv_data = rfmt_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="⬇️ Unduh Report CSV", data=csv_data, file_name=f"Churn_Report_{window_selection}.csv", mime="text/csv", type="primary")

        except Exception as e:
            st.error(f"Gagal memproses file. Pastikan struktur CSV benar. Detail Error: {e}")
