# 💬 Twitter Sentiment Analysis

## 📌 Project Description
This project is a complete **Twitter Sentiment Analysis pipeline** that predicts whether a tweet is **Positive**, **Neutral**, or **Negative**.  
It covers everything from **data preprocessing**, **exploratory analysis**, **model training**, to **deploying a Streamlit web app** for real-time predictions.

The dataset used is `Tweets.csv`, which contains 27,481 tweets with their text and sentiment labels.

---
## [🔗 **Live Streamlit App**](https://twitter-sentiment-analysis1.streamlit.app/)


---

## 📝 Project Workflow

### 1️⃣ Data Preprocessing
Before training the models, extensive preprocessing was applied to clean and normalize the text:

- **Removed unnecessary columns** (`textID`) and handled missing values.
- **Dropped duplicates** to avoid redundant data.
- **Mapped sentiment labels to numeric categories**:
  - Neutral → 0  
  - Negative → 1  
  - Positive → 2
- **Text cleaning included**:
  - Converting all text to lowercase
  - Removing URLs, mentions (@username), and hashtag symbols (#)
  - Removing punctuation, numbers, and special characters
  - Reducing repeated letters (e.g., `cooool` → `cool`)
  - Removing extra spaces
  - Tokenization, stopwords removal, and lemmatization (using NLTK)
- **Result:** A new `clean_text` column with normalized text ready for feature extraction

---

### 2️⃣ Exploratory Data Analysis
- Created **WordCloud** to visualize the most frequent words in tweets.

  <img width="790" height="429" alt="image" src="https://github.com/user-attachments/assets/83e10da7-5a0a-4e92-8cff-b1107bf35fad" />

- Analyzed sentiment distribution:
  - Neutral: 40.5%  
  - Positive: 31.2%  
  - Negative: 28.3%

---

### 3️⃣ Feature Extraction
- Converted text into numerical features using **TF-IDF Vectorization**.
- Considered unigrams, bigrams, and trigrams.

---

### 4️⃣ Handling Class Imbalance
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.
- After SMOTE, all classes have equal number of samples.

---

### 5️⃣ Model Training
Multiple models were trained for sentiment classification:

| Model                  | Test Accuracy |
|------------------------|---------------|
| Linear SVC             | 71.6%        |
| Logistic Regression    | 72.3%        |
| Random Forest          | 75.2%        |
| XGBoost Classifier     | 71.1%        |

**Chosen model for deployment:** **XGBoost Classifier**

---

### 6️⃣ Model Evaluation
- Test accuracy: 71.1%
- Weighted F1-score: 0.71
- Model shows balanced performance across all sentiment classes.

---

### 7️⃣ Streamlit App
- Provides an interactive interface for predicting sentiment from user input.
- Applies the same preprocessing as in the notebook.
- Displays prediction in a colored box:
  - Neutral 😐 → Light Yellow
  - Negative 😠 → Light Red
  - Positive 😊 → Light Green
- User-friendly and visually appealing interface.

---

## 📂 Project Files
├── Tweets.csv # Original dataset
├── sentiment_analysis.ipynb # Jupyter Notebook with preprocessing, EDA, and model training
├── xgb_model.joblib # Trained XGBoost model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── app.py # Streamlit web app
└── README.md # Project documentation


---

## 🛠️ Libraries & Tools Used
- Python 3
- Pandas, NumPy
- NLTK (Stopwords, WordNet, POS tagging, Lemmatization)
- Scikit-learn (TF-IDF, SVM, Logistic Regression, Random Forest, Train/Test Split)
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn, WordCloud
- Streamlit (Web App)
- Joblib (Model & Vectorizer Serialization)

---

## 📈 Conclusion
This project demonstrates a **full NLP pipeline** for Twitter sentiment analysis:  
From data cleaning and visualization to model training, balancing classes with SMOTE, and deploying a **real-time web app**.  

The app can be used for **social media monitoring, brand analysis, and public opi
