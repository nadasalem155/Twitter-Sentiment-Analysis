# =============================
# 🎯 Sentiment Analysis App with Streamlit
# =============================
import nltk
import streamlit as st
import re
import joblib
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# NLTK data path
nltk_data_path = '/home/appuser/nltk_data'
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)  

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
    tweet = str(tweet).lower()
    tweet = re.sub(r"http\S+|www.\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tweet = re.sub(r"[^a-z\s]", " ", tweet)
    tweet = re.sub(r"(.)\1{2,}", r"\1", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    words = tweet.split()
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stop_words]
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

user_input = st.text_area("✏️ Enter your text here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analysis.")
    else:
        clean_text = clean_tweet(user_input)
        text_vector = vectorizer.transform([clean_text])
        prediction = model.predict(text_vector)[0]
        sentiment_map = {0: ("Neutral 😐", "#FFFF00"), 1: ("Negative 😠", "#FF0000"), 2: ("Positive 😊", "#00FF00")}
        sentiment_label, bg_color = sentiment_map.get(prediction, ("Unknown", "#FFFFFF"))
        
        st.subheader("🧾 Result:")
        st.markdown(
            f"<div style='background-color:{bg_color}; color:#000000; padding:15px; border-radius:10px; font-size:22px;'>"
            f"<b>{sentiment_label}</b></div>", unsafe_allow_html=True
        )

st.markdown("---")