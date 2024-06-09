import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Memuat data dan model yang telah diproses
@st.cache(allow_output_mutation=True)
def load_resources():
    data = pd.read_csv("processed_reviews.csv")
    model = joblib.load('model_pipeline.pkl')
    return data, model

def visualize_sentiment_predictions(predictions):
    sentiment_counts = pd.Series(predictions).value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index.astype(str), sentiment_counts.values)
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah')
    ax.set_title('Hasil Prediksi Sentimen')
    st.pyplot(fig)

df, model_pipeline = load_resources()

# Streamlit interface
st.title("Analisis Sentimen Ulasan Hotel")
review_input = st.text_area("Tulis ulasan:")
if st.button("Prediksi Sentimen") and review_input:
    prediction = model_pipeline.predict([review_input])[0]
    result = "Positif" if prediction == 1 else "Negatif"
    st.success(f"Sentimen yang diprediksi adalah: {result}")

uploaded_file = st.file_uploader("Unggah file CSV berisi ulasan", type=["csv"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(uploaded_df.head())
    if st.button("Prediksi Sentimen dari File"):
        predictions = model_pipeline.predict(uploaded_df['Review'])
        uploaded_df['predicted_sentiment'] = predictions
        st.write("Hasil Prediksi:")
        st.write(uploaded_df)
        visualize_sentiment_predictions(predictions)
