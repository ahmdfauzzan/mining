import pandas as pd
import re
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load and prepare data
@st.cache_data  # Updated cache function
def load_data():
    df = pd.read_csv("gabungan-semua.csv", encoding="latin-1")
    df['Rating'] = df['Rating'].apply(lambda x: x.replace(',', '').replace('/5', '').strip()).astype(float).astype(int)
    df.loc[df['Rating'] == 41, 'Rating'] = 4
    return df

df = load_data()

# Assign sentiment
df['sentiment'] = df['Rating'].apply(lambda rating: 1 if rating > 3 else -1)

# Clean and tokenize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https\S+|@\S+|#\S+|\'\w+|[^\w\s]|', '', text)  # Improved regex
    text = re.sub(r'\s2\s', ' ', text)
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)
    return tokens

df['content_token'] = df['Review'].apply(clean_text)

# Remove stopwords and perform stemming
def remove_stopwords_and_stem(tokens):
    stopwords_list = nltk.corpus.stopwords.words("indonesian")
    tokens = [word for word in tokens if word not in stopwords_list]
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['text_processed'] = df['content_token'].apply(remove_stopwords_and_stem)

# Vectorize text
tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
X = tfidf.fit_transform(df['text_processed'])
y = df['sentiment']

# Handle imbalanced dataset
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.1, random_state=3)

# Train SVM
svm = SVC()
svm.fit(X_train, y_train)

# Streamlit interface
st.title("Analisis Sentimen Ulasan Hotel")

review_input = st.text_area("Tulis ulasan:")

if st.button("Prediksi Sentimen"):
    if review_input:
        tokens = clean_text(review_input)
        preprocessed_text = remove_stopwords_and_stem(tokens)
        input_vect = tfidf.transform([preprocessed_text])
        prediction = svm.predict(input_vect)
        result = "Positif" if prediction[0] == 1 else "Negatif"
        st.success(f"Sentimen yang diprediksi adalah: {result}")
    else:
        st.error("Silakan masukkan ulasan untuk diprediksi.")
