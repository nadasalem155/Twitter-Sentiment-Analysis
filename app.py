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

# =============================
# 📥 NLTK Resources
# =============================
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
# 🎨 Streamlit Interface Styling
# =============================
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
h1 {
    color: #4b6cb7;
    text-align: center;
}
.stButton>button {
    background-color: #4b6cb7;
    color: white;
    font-weight: bold;
    height: 3em;
    width: 100%;
    border-radius: 10px;
}
.stTextArea>div>div>textarea {
    border-radius: 10px;
    border: 1px solid #ccc;
}
.result-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Sentiment Analysis App")
st.write("Enter a tweet or any text below to predict its sentiment (Positive / Neutral / Negative).")

user_input = st.text_area("✏️ Enter your text here:")

# =============================
# 🔍 Prediction
# =============================
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analysis.")
    else:
        clean_text = clean_tweet(user_input)
        text_vector = vectorizer.transform([clean_text])
        prediction = model.predict(text_vector)[0]
        sentiment_map = {0: "Neutral 😐", 1: "Negative 😠", 2: "Positive 😊"}
        sentiment_label = sentiment_map.get(prediction, "Unknown")

        # Display in a card
        st.markdown(f"""
        <div class="result-card">
            <h3>🧾 Result:</h3>
            <p style="font-size: 1.3em;">The sentiment is: <b>{sentiment_label}</b></p>
        </div>
        """, unsafe_allow_html=True)

# =============================
# 🧠 Footer
# =============================
st.markdown("---")
