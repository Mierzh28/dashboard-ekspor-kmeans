import streamlit as st
import pandas as pd

# Set page layout to wide
st.set_page_config(layout="wide")

# Sidebar - File Upload
st.sidebar.title("Unggah Data")
st.sidebar.info("Unggah file CSV atau Excel untuk analisis clustering")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Memeriksa nama kolom untuk memastikan kolom yang benar
    st.write("Nama kolom yang tersedia dalam dataset:")
    st.write(df.columns)

    # Menampilkan data yang diunggah
    st.title("Data Ekspor")
    st.write("Berikut adalah data yang diunggah:")
    st.dataframe(df.head())

    # Memastikan kolom yang digunakan ada dan valid
    if 'Nama Perusahaan' in df.columns:
        transaksi_perusahaan = df.groupby('Nama Perusahaan').size().reset_index(name='Jumlah Transaksi')
        transaksi_perusahaan_sorted = transaksi_perusahaan.sort_values(by='Jumlah Transaksi', ascending=False)
        
        st.write("Berikut adalah perusahaan yang sering melakukan transaksi, diurutkan berdasarkan jumlah transaksi terbanyak:")
        st.dataframe(transaksi_perusahaan_sorted)
    else:
        st.write("Kolom 'Nama Perusahaan' tidak ditemukan dalam data. Periksa kembali dataset Anda.")
    
else:
    st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")
