import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
from sklearn.preprocessing import LabelEncoder

# 1. Sayfa Ayarları
st.set_page_config(page_title="Araç Fiyat Tahmin Sistemi", page_icon="🚗", layout="centered")

st.title("🚗 Avustralya Araç Fiyat Tahmin Motoru")
st.write("Listeden aracınızın markasını, modelini ve özelliklerini seçerek anlık piyasa değerini hesaplayın.")
st.markdown("---")

# Dinamik klasör yolu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def veriyi_ve_encoderlari_hazirla():
    zip_dosya_adi = "arabalar.zip" 
    zip_file_path = os.path.join(BASE_DIR, zip_dosya_adi)
    
    if not os.path.exists(zip_file_path):
        st.error(f"❌ '{zip_dosya_adi}' dosyası GitHub deponuzda bulunamadı!")
        st.stop()
        
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            st.error("❌ Zip dosyasının içinde .csv uzantılı bir veri dosyası bulunamadı!")
            st.stop()
        
        with z.open(csv_files[0]) as f:
            df_temiz = pd.read_csv(f)

    # 🎯 KİLOMETRE VE DİĞER SAYISAL SÜTUNLARI METİNDEN ARINDIRMA ROBOTU
    # Eğer sütunda "15,000 km" veya "2.0L" gibi metinler varsa sadece sayıları ayıklar
    sayisal_sutunlar = ['Kilometres', 'Engine', 'FuelConsumption', 'Year', 'Price']
    for col in sayisal_sutunlar:
        if col in df_temiz.columns:
            # Virgülleri kaldır ve stringe çevir
            df_temiz[col] = df_temiz[col].astype(str).str.replace(',', '', regex=True)
            # Sadece sayısal kısımları (regex ile) ayıkla
            df_temiz[col] = df_temiz[col].str.extract(r'(\d+\.?\d*)')[0]
            # Gerçekten sayısal veri tipine (float/int) dönüştür
            df_temiz[col] = pd.to_numeric(df_temiz[col], errors='coerce')
            # Eğer boş (NaN) satırlar kalırsa sıfırla doldur ki hata vermesin
            df_temiz[col] = df_temiz[col].fillna(0)

    marka_listesi = sorted(df_temiz['Brand'].dropna().unique())
    vites_listesi = sorted(df_temiz['Transmission'].dropna().unique())
    yakit_listesi = sorted(df_temiz['FuelType'].dropna().unique())
    tip_listesi = sorted(df_temiz['Type'].dropna().unique()) if 'Type' in df_temiz.columns else []

    encoders_dict = {}
    for col in ['Brand', 'Model', 'Transmission', 'FuelType']:
        if col in df_temiz.columns:
            le = LabelEncoder()
            le.fit(df_temiz[col].astype(str))
            encoders_dict[col] = le

    return marka_listesi, vites_listesi, yakit_listesi, tip_listesi, encoders_dict, df_temiz

# Verileri çekiyoruz
marka_listesi, vites_listesi, yakit_listesi, tip_listesi, encoders_dict, df_temiz = veriyi_ve_encoderlari_hazirla()

# Model ve Scaler yükleme
model_path = os.path.join(BASE_DIR, 'en_iyi_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# 2. Kullanıcı Arayüzü (UI)
st.subheader("📋 Araç Özelliklerini Seçiniz")

col1, col2 = st.columns(2)

with col1:
    secilen_marka = st.selectbox("Araç Markası (Brand)", options=marka_listesi)

    modeller_filtreli = sorted(df_temiz[df_temiz['Brand'] == secilen_marka]['Model'].unique())
    secilen_model = st.selectbox("Araç Modeli (Model)", options=modeller_filtreli)

    yil = st.slider("Model Yılı (Year)", min_value=2000, max_value=2026, value=2018)
    kilometre = st.number_input("Kilometre (KM)", min_value=0, value=65000, step=1000)

with col2:
    secilen_vites = st.selectbox("Vites Tipi (Transmission)", options=vites_listesi)
    secilen_yakit = st.selectbox("Yakıt Türü (Fuel Type)", options=yakit_listesi)
    motor_hacmi = st.number_input("Motor Hacmi (Litre - Engine)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    yakit_tuketimi = st.slider("Yakıt Tüketimi (L/100 km)", min_value=0.0, max_value=25.0, value=7.2, step=0.1)

st.markdown("---")

# 3. Tahmin Motoru Tetikleyicisi
if st.button("💰 Aracın Fiyatını Hesapla", use_container_width=True):
    try:
        orijinal_sutunlar = [col for col in df_temiz.columns if col != 'Price']

        kodlu_girdi = {}
        for col in orijinal_sutunlar:
            try:
                kodlu_girdi[col] = df_temiz[col].mean()
            except Exception:
                kodlu_girdi[col] = 0

        kodlu_girdi['Brand'] = encoders_dict['Brand'].transform([secilen_marka])[0]

        try:
            kodlu_girdi['Model'] = encoders_dict['Model'].transform([secilen_model])[0]
        except Exception:
            kodlu_girdi['Model'] = 0

        kodlu_girdi['Transmission'] = encoders_dict['Transmission'].transform([secilen_vites])[0]
        kodlu_girdi['FuelType'] = encoders_dict['FuelType'].transform([secilen_yakit])[0]

        kodlu_girdi['Year'] = yil
        
        # Artık Kilometres sütunu kesinlikle sayı olduğu için bu formül asla patlamaz!
        ort_km = df_temiz['Kilometres'].mean()
        ters_kilometre = (2 * ort_km) - kilometre
        kodlu_girdi['Kilometres'] = max(0, ters_kilometre)

        kodlu_girdi['Engine'] = motor_hacmi
        kodlu_girdi['FuelConsumption'] = yakit_tuketimi

        for col in encoders_dict.keys():
            if col in kodlu_girdi and col not in ['Brand', 'Model', 'Transmission', 'FuelType']:
                kodlu_girdi[col] = 0

        girdi_df = pd.DataFrame([kodlu_girdi])[orijinal_sutunlar]
        girdi_olcekli = scaler.transform(girdi_df)
        tahmin_fiyat = model.predict(girdi_olcekli)[0]

        if tahmin_fiyat < 0:
            tahmin_fiyat = df_temiz['Price'].median() * 0.5

        st.success(f"### 🎯 Tahmin Edilen Değer: ${tahmin_fiyat:,.2f} AUD")
        st.caption(f"Seçilen Segment: {secilen_marka} - {secilen_model} ({yil})")

    except Exception as e:
        st.error(f"Tahmin motorunda bir uyumsuzluk yaşandı: {e}")
