import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

# Memuat data dan model yang telah diproses
@st.cache(allow_output_mutation=True)
def load_resources():
    data = pd.read_csv("processed_reviews.csv")
    model = joblib.load('model_pipeline.pkl')
    return data, model

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return filtered_tokens

def visualize_sentiment_predictions(predictions):
    sentiment_counts = pd.Series(predictions).value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index.astype(str), sentiment_counts.values)
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah')
    ax.set_title('Hasil Prediksi Sentimen')
    st.pyplot(fig)

def analyze_reviews(uploaded_df):
    predictions = model_pipeline.predict(uploaded_df['Review'])
    uploaded_df['predicted_sentiment'] = predictions

    # Hitung total sentimen positif dan negatif
    sentiment_counts = pd.Series(predictions).value_counts()
    total_positive = sentiment_counts[1] if 1 in sentiment_counts else 0
    total_negative = sentiment_counts[-1] if -1 in sentiment_counts else 0

    st.write("Total Sentimen Positif:", total_positive)
    st.write("Total Sentimen Negatif:", total_negative)

    # Ambil kata-kata yang paling sering muncul di sentimen positif dan negatif
    positive_words = []
    negative_words = []

    for index, row in uploaded_df.iterrows():
        if row['predicted_sentiment'] == 1:
            positive_words.extend(preprocess_text(row['Review']))
        else:
            negative_words.extend(preprocess_text(row['Review']))

    most_common_positive = Counter(positive_words).most_common(5)
    most_common_negative = Counter(negative_words).most_common(5)

    st.write("Kata-kata Paling Muncul dalam Sentimen Positif:")
    st.write([word[0] for word in most_common_positive])

    st.write("Kata-kata Paling Muncul dalam Sentimen Negatif:")
    st.write([word[0] for word in most_common_negative])

    st.write("Hasil Prediksi:")
    st.write(uploaded_df)
    visualize_sentiment_predictions(predictions)

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
        analyze_reviews(uploaded_df)
