from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
import re

# Pre-download NLTK data
nltk.download('stopwords')

# Load and prepare data
@st.experimental_singleton
def load_data():
    df = pd.read_csv("gabungan-semua.csv", encoding="latin-1")
    df.drop(columns=['Name', 'Date'], inplace=True)
    df['cleaned_text'] = df['Review'].apply(lambda x: re.sub('[^a-zA-Z\s]', ' ', x).lower().strip())
    
    # Konversi kolom 'Rating' menjadi numerik
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Membuat kolom 'label' dengan mengasumsikan bahwa 'NaN' di 'Rating' akan menghasilkan nilai False
    df['label'] = df['Rating'].apply(lambda x: 1 if x > 3 else 0)
    
    return df

df = load_data()

# Tokenization and Stemming using Sastrawi
def process_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    words = nltk.word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 1]
    return " ".join(stemmed)

# Model Training
@st.experimental_singleton
def train_model():
    tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
    X = df['cleaned_text'].apply(process_text)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    model = SVC(kernel='linear', random_state=10)
    model.fit(X_train_tfidf, y_train)
    return tfidf, model

tfidf, model = train_model()

# Streamlit Interface
st.title("Analisis Sentimen Ulasan Hotel")

review_input = st.text_area("Tulis ulasan:")

if st.button("Prediksi Sentimen"):
    if review_input:
        processed_input = process_text(review_input)
        input_vect = tfidf.transform([processed_input])
        prediction = model.predict(input_vect)
        result = "Positif" if prediction[0] == 1 else "Negatif"
        st.success(f"Sentimen yang diprediksi adalah: {result}")
    else:
        st.error("Silakan masukkan ulasan untuk diprediksi.")
