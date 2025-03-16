import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_data,
    plot_sentiment_distribution,
    train_model,
    train_model_split_80_20,
    train_model_split_70_30,
    train_model_split_60_40,
    plot_confusion_matrix,
    plot_classification_metrics,
    text_preprocessing,
    full_text_pipeline,
    load_normalization_dict
)

# Load data from CSV file
data = load_data("data/6Fixpreprocecing - Fixpreprocecing.csv")

# Handle missing values in 'full_text' column
data['full_text'] = data['full_text'].astype(str)

# Load normalization dictionary (pastikan file kamuskatabaku.csv tersedia di folder data)
kamus_tidak_baku = load_normalization_dict("data/kamuskatabaku.csv")

# Preprocess data menggunakan full pipeline (tanpa visual tambahan)
# Hasil akhir disimpan di kolom 'cleaning' agar tetap kompatibel dengan fungsi pelatihan
data['cleaning'] = data['full_text'].apply(lambda x: full_text_pipeline(x, kamus_tidak_baku))

# Remove rows with missing values in 'cleaning' or 'sentimen'
data = data.dropna(subset=['cleaning', 'sentimen'])

# Check class distribution
class_counts = data['sentimen'].value_counts()
print("Class distribution:", class_counts)

# Remove classes with less than 2 samples
min_samples = 2
data = data[data['sentimen'].map(class_counts) >= min_samples]

# Set page config
st.set_page_config(page_title="ANALISIS SENTIMEN", layout="wide")

# Custom CSS (tetap sama, tanpa perubahan visual)
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding: 0;
    }
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 50px;
        background-color: dark;
        color: white;
    }
    .logo {
        font-size: 28px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .nav ul {
        display: flex;
        list-style: none;
    }
    .nav ul li {
        margin-left: 25px;
    }
    .nav ul li a {
        color: white;
        text-decoration: none;
        font-size: 16px;
        padding: 5px 0;
        position: relative;
    }
    .nav ul li a:hover {
        color: #ccc;
    }
    .nav ul li a.active {
        color: white;
        border-bottom: 2px solid white;
    }
    .main-content {
        color: white;
        padding: 30px 50px;
    }
    .dashboard-title {
        text-align: center;
        font-size: 60px;
        margin: -90px 0 20px 0;
        font-weight: normal;
        letter-spacing: 2px;
    }
    .content-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 50px;
    }
    .text-section {
        flex: 0 0 60%;
    }
    .title-section p {
        color: #aaa;
        margin-bottom: 20px;
        font-size: 16px;
    }
    .thesis-title {
        font-size: 24px;
        text-transform: uppercase;
        line-height: 1.5;
        letter-spacing: 1px;
    }
    .image-section {
        flex: 0 0 35%;
        text-align: center;
    }
    .image-wrapper img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        border-radius: 5px;
    }
    .profile-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .name {
        font-size: 20px;
        margin-bottom: 20px;
        color: #aaa;
    }
    .id-number {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .footer {
        background-color: white;
        padding: 20px 50px;
        text-align: left;
        color: #333;
    }
    .footer-name {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Dataset", "Analisis Teks", "Model Evaluasi"], index=0)

if page == "Dashboard":
    st.markdown('<div class="header"><div class="logo">ANALISIS SENTIMEN</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown('<h1 class="dashboard-title">Dashboard</h1>', unsafe_allow_html=True)
        col1, col2 = st.columns([6, 4])
        with col1:
            st.markdown('<div class="text-section">', unsafe_allow_html=True)
            st.markdown('<div class="title-section">', unsafe_allow_html=True)
            st.markdown('<p>Judul Skripsi</p>', unsafe_allow_html=True)
            st.markdown('<h2 class="thesis-title">ANALISIS SENTIMEN MASYARAKAT TERHADAP KENAIKAN PPN MENJADI 12% DI INDONESIA PADA MEDIA SOSIAL X MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE</h2>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="image-section">', unsafe_allow_html=True)
            st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
            st.image("C:/Users/ahmdf/Pictures/Screenshots/foto profil.png")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p class="name">Octafiana Hanani Fityati Syifa</p>', unsafe_allow_html=True)
            st.markdown('<p class="id-number">21.12.2158</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<footer class="footer"><p class="footer-name">Octafiana Hanani Fityati Syifa</p></footer>', unsafe_allow_html=True)

elif page == "Dataset":
    st.header("Dataset Data Sentimen")
    st.subheader("Data Mentah")
    st.dataframe(data)
    st.subheader("Distribusi Sentimen")
    sentiment_counts = data['sentimen'].value_counts()
    fig = plot_sentiment_distribution(sentiment_counts, figsize=(6, 4))
    st.pyplot(fig)

elif page == "Analisis Teks":
    st.header("Analisis Teks")
    text_input = st.text_area("Masukkan teks untuk dianalisis")
    if st.button("Analisis"):
        st.write("Hasil analisis akan ditampilkan di sini")

elif page == "Model Evaluasi":
    st.header("Evaluasi Model")
    # Split ratios: 0.1 = 90:10, 0.2 = 80:20, 0.3 = 70:30, 0.4 = 60:40
    split_ratios = [0.1, 0.2, 0.3, 0.4]
    split_labels = ["90:10", "80:20", "70:30", "60:40"]
    cols = st.columns(len(split_ratios), gap="small")
    
    for i, (ratio, label) in enumerate(zip(split_ratios, split_labels)):
        with cols[i]:
            st.subheader(f"Pemisahan Data {label}")
            # Panggil fungsi pelatihan sesuai rasio
            if ratio == 0.1:
                # Split 90:10
                accuracy, report, y_test, y_pred = train_model(data, test_size=ratio)
            elif ratio == 0.2:
                # Split 80:20
                accuracy, report, y_test, y_pred = train_model_split_80_20(data, test_size=ratio)
            elif ratio == 0.3:
                # Split 70:30
                accuracy, report, y_test, y_pred = train_model_split_70_30(data, test_size=ratio)
            elif ratio == 0.4:
                # Split 60:40
                accuracy, report, y_test, y_pred = train_model_split_60_40(data, test_size=ratio)
            
            st.write(f"Akurasi Model: {accuracy:.2%}")
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig_cm)
            st.subheader("Metrik Klasifikasi")
            fig_metrics = plot_classification_metrics(report)
            st.pyplot(fig_metrics)
