# =============================
# 🎯 Sentiment Analysis App with Streamlit
# =============================
# =============================
# 📥 Download NLTK Resources (fix for server)
# =============================
import nltk

nltk_data_path = '/home/appuser/nltk_data'
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

import streamlit as st
import re
import nltk
import joblib
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# =============================
# 🧩 Helper Function: POS Tag Mapping
# =============================
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# =============================
# 🧹 Text Cleaning Function
# =============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_tweet(tweet):
    # Convert to lowercase
    tweet = str(tweet).lower()

    # Remove URLs
    tweet = re.sub(r"http\S+|www.\S+", "", tweet)

    # Remove mentions (@username)
    tweet = re.sub(r"@\w+", "", tweet)

    # Remove hashtags symbol (#)
    tweet = re.sub(r"#", "", tweet)

    # Remove punctuation and numbers
    tweet = re.sub(r"[^a-z\s]", " ", tweet)

    # Remove repeated letters (e.g. cooool → cool)
    tweet = re.sub(r"(.)\1{2,}", r"\1", tweet)

    # Remove extra spaces
    tweet = re.sub(r"\s+", " ", tweet).strip()

    # Tokenize, remove stopwords, and lemmatize
    words = tweet.split()
    words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in words if word not in stop_words
    ]

    return " ".join(words)

# =============================
# 📦 Load Model and Vectorizer
# =============================
model = joblib.load("xgb_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  

# =============================
# 🎨 Streamlit User Interface
# =============================
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis App")
st.write("Enter a tweet or any text below to predict its sentiment (Positive / Neutral / Negative).")

# Text input area
user_input = st.text_area("✏️ Enter your text here:")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analysis.")
    else:
        # Clean the input text
        clean_text = clean_tweet(user_input)

        # Transform text using the TF-IDF vectorizer
        text_vector = vectorizer.transform([clean_text])

        # Predict sentiment
        prediction = model.predict(text_vector)[0]

        # Map numeric prediction back to label
        sentiment_map = {0: "Neutral 😐", 1: "Negative 😠", 2: "Positive 😊"}
        sentiment_label = sentiment_map.get(prediction, "Unknown")

        # Display prediction result
        st.subheader("🧾 Result:")
        st.success(f"The sentiment is: **{sentiment_label}**")

# =============================
# 🧠 Footer
# =============================
st.markdown("---")
