import streamlit as st
import pandas as pd
import joblib

# Memuat data dan model yang telah diproses
@st.cache(allow_output_mutation=True)
def load_resources():
    data = pd.read_csv("processed_reviews.csv")
    model = joblib.load('model_pipeline.pkl')
    return data, model

df, model_pipeline = load_resources()

# Streamlit interface
st.title("Analisis Sentimen Ulasan Hotel")
review_input = st.text_area("Tulis ulasan:")
if st.button("Prediksi Sentimen") and review_input:
    prediction = model_pipeline.predict([review_input])[0]
    result = "Positif" if prediction == 1 else "Negatif"
    st.success(f"Sentimen yang diprediksi adalah: {result}")
else:
    st.error("Silakan masukkan ulasan untuk diprediksi.")
