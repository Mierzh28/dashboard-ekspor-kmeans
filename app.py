import streamlit as st
import pandas as pd
import numpy as np

# Mengunggah file
uploaded_file = st.sidebar.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Memuat data
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Menampilkan 5 baris pertama untuk memeriksa struktur data
        st.write("Berikut adalah data yang diunggah:")
        st.dataframe(df.head())

        # Menampilkan informasi tipe data kolom
        st.markdown("### Tipe Data Setiap Kolom")
        st.write(df.dtypes)

        # Periksa apakah kolom yang digunakan ada dan bertipe numerik
        if 'FOB_USD' not in df.columns or 'Qty' not in df.columns:
            st.error("Kolom 'FOB_USD' atau 'Qty' tidak ditemukan dalam data. Silakan periksa file Anda.")
        else:
            # Cek data yang ada pada kolom yang relevan
            st.markdown("### Memeriksa Nilai Unik pada Kolom 'FOB_USD' dan 'Qty'")
            st.write(f"Nilai unik di kolom 'FOB_USD': {df['FOB_USD'].unique()}")
            st.write(f"Nilai unik di kolom 'Qty': {df['Qty'].unique()}")

            # Mengonversi kolom menjadi numerik dan menangani nilai yang tidak valid
            df['FOB_USD'] = pd.to_numeric(df['FOB_USD'], errors='coerce')
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

            # Memeriksa apakah ada nilai NaN setelah konversi
            st.markdown("### Nilai NaN Setelah Konversi ke Numerik")
            st.write(f"Jumlah NaN di kolom 'FOB_USD': {df['FOB_USD'].isna().sum()}")
            st.write(f"Jumlah NaN di kolom 'Qty': {df['Qty'].isna().sum()}")

            # Mengganti NaN dengan rata-rata kolom
            df['FOB_USD'].fillna(df['FOB_USD'].mean(), inplace=True)
            df['Qty'].fillna(df['Qty'].mean(), inplace=True)

            # Menghapus baris yang masih kosong setelah pengisian NaN
            df_clean = df[['FOB_USD', 'Qty']].dropna()

            if df_clean.empty:
                st.error("Data yang Anda unggah tidak memiliki data yang valid untuk analisis. Pastikan semua data numerik terisi.")
            else:
                st.success("Data siap untuk analisis clustering!")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
else:
    st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")
