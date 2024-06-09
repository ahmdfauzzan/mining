import streamlit as st
import pandas as pd
import joblib
import speech_recognition as sr

# Memuat data dan model yang telah diproses
@st.cache(allow_output_mutation=True)
def load_resources():
    data = pd.read_csv("processed_reviews.csv")
    model = joblib.load('model_pipeline.pkl')
    return data, model

df, model_pipeline = load_resources()

# Fungsi untuk merekam suara dan mengonversi menjadi teks
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Mendengarkan... Silakan bicara.")
        audio = recognizer.listen(source)
    try:
        st.info("Mengonversi suara ke teks...")
        text = recognizer.recognize_google(audio, language='id-ID')  # Menggunakan bahasa Indonesia
        st.success("Selesai merekam!")
        return text
    except sr.UnknownValueError:
        st.error("Tidak dapat mengenali suara. Silakan coba lagi.")
        return ""
    except sr.RequestError as e:
        st.error(f"Permintaan ke layanan pengenalan suara gagal; {e}")
        return ""

# Streamlit interface
st.title("Analisis Sentimen Ulasan Hotel")
review_input = st.text_area("Tulis ulasan:")
if st.button("Rekam Ulasan"):
    review_input = recognize_speech()
    st.text_area("Ulasan dari suara:", review_input)

if st.button("Prediksi Sentimen") and review_input:
    prediction = model_pipeline.predict([review_input])[0]
    result = "Positif" if prediction == 1 else "Negatif"
    st.success(f"Sentimen yang diprediksi adalah: {result}")
else:
    st.error("Silakan masukkan ulasan untuk diprediksi.")

