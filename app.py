import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Araç Fiyat Tahmin Sistemi", page_icon="🚗", layout="centered")

st.title("🚗 Avustralya Araç Fiyat Tahmin Motoru")
st.write("Listeden aracınızın markasını, modelini ve özelliklerini seçerek anlık piyasa değerini hesaplayın.")
st.markdown("---")

@st.cache_data
# @st.cache_data veya hangisini kullanıyorsan o kalsın
def veriyi_ve_encoderlari_hazirla():
    # 1. Klasör yolunu dinamik yapıyoruz (Sunucuda sorun çıkarmaması için en garanti yol)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. ZIP DOSYASININ ADINI BURAYA YAZ (Örn: 'veriler.zip' veya klasör içindeyse 'data/veri.zip')
    # GitHub'daki dosya adıyla BİREBİR aynı olmalı (Büyük/küçük harfe duyarlıdır!)
    zip_dosya_adi = "verileriniz.zip" 
    
    zip_file_path = os.path.join(current_dir, zip_dosya_adi)
    
    # 3. Güvenlik Duvarı: Eğer dosya yine bulunamazsa uygulama çökmek yerine sana söylesin
    if not os.path.exists(zip_file_path):
        st.error(f"❌ Zip dosyası bulunamadı!")
        st.warning(f"Sistem şu adrese baktı ama bulamadı: {zip_file_path}")
        st.info("Lütfen aşağıdaki 3 maddelik kontrol listesine göz at.")
        st.stop() # Uygulamanın çökmesini engeller, burada durdurur.

    # 4. Dosya varsa sorunsuz açılacaktır
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # --- BURADAN SONRASI SENİN KENDİ KODLARIN ---
        # Dosyayı z.extractall() vb. şeklinde okuduğun kısım...
        pass
        
    return marka_listesi, vites_listesi, yakit_listesi, tip_listesi, encoders_dict, df_temiz

with open('en_iyi_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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

if st.button("💰 Aracın Fiyatını Hesapla", use_container_width=True):
    try:
        orijinal_sutunlar = [col for col in df_temiz.columns if col != 'Price']

        kodlu_girdi = {}
        for col in orijinal_sutunlar:
            if df_temiz[col].dtype == 'object':
                kodlu_girdi[col] = 0
            else:
                kodlu_girdi[col] = df_temiz[col].mean()

        kodlu_girdi['Brand'] = encoders_dict['Brand'].transform([secilen_marka])[0]

        try:
            kodlu_girdi['Model'] = encoders_dict['Model'].transform([secilen_model])[0]
        except:
            kodlu_girdi['Model'] = 0

        kodlu_girdi['Transmission'] = encoders_dict['Transmission'].transform([secilen_vites])[0]
        kodlu_girdi['FuelType'] = encoders_dict['FuelType'].transform([secilen_yakit])[0]

        kodlu_girdi['Year'] = yil
        kodlu_girdi['Kilometres'] = kilometre
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
