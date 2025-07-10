import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("bias_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Stopwords for cleaning
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("📰 Bias Language Detector in Articles")
user_input = st.text_area("Paste your article here:")

if st.button("Predict Bias"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("⚠️ This article may contain **Biased** language.")
        else:
            st.success("✅ This article appears to be **Unbiased**.")
