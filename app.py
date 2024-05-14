import pandas as pd
import re
import streamlit as st
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

nltk.download('stopwords')

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("gabungan-semua.csv", encoding="latin-1")
    df['Rating'] = df['Rating'].apply(lambda x: x.replace(',', '').replace('/5', '').strip()).astype(float).astype(int)
    df.loc[df['Rating'] == 41, 'Rating'] = 4
    df['Sentiment'] = df['Rating'].apply(lambda rating: 1 if rating > 3 else -1)
    return df

df = load_data()

def clean_and_stem(text):
    text = text.lower()
    text = re.sub(r'https\S+|@\S+|#\S+|\'\w+|[^\w\s]|', '', text)
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)
    stopwords_list = stopwords.words("indonesian")
    tokens = [word for word in tokens if word not in stopwords_list]
    stemmer = StemmerFactory().create_stemmer()
    return ' '.join(stemmer.stem(token) for token in tokens)

# Creating a pipeline
model_pipeline = make_pipeline_imb(
    TfidfVectorizer(max_df=0.5, min_df=2, preprocessor=clean_and_stem),
    SMOTE(),
    SVC()
)

X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.1, random_state=3)
model_pipeline.fit(X_train, y_train)

# Streamlit interface
st.title("Analisis Sentimen Ulasan Hotel")
review_input = st.text_area("Tulis ulasan:")
if st.button("Prediksi Sentimen"):
    if review_input:
        prediction = model_pipeline.predict([review_input])[0]
        result = "Positif" if prediction == 1 else "Negatif"
        st.success(f"Sentimen yang diprediksi adalah: {result}")
    else:
        st.error("Silakan masukkan ulasan untuk diprediksi.")
