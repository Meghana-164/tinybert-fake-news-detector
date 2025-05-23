import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and tokenizer
model = joblib.load("bert_tiny_logreg_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("bert_tiny_tokenizer/")
bert_model = AutoModel.from_pretrained("bert_tiny_model/")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return " ".join([word for word in text.split() if word not in stop_words])

# BERT embedding function
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)

# Prediction function
def predict_fake_news(text):
    cleaned = clean_text(text)
    embedding = get_bert_embedding(cleaned)
    prediction = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0]
    confidence = prob[prediction] * 100
    return prediction, confidence

# Streamlit GUI
st.set_page_config(page_title="FAKE NEWS DETECTOR", page_icon="📰")
st.title("📰 Fake News Detector ")
st.write("Enter a news headline or article to classify it as **real** or **fake**.")

user_input = st.text_area("News Text :", height=200, placeholder="Type or paste news content here...")

if st.button("🔍 Detect"):
    if user_input.strip():
        prediction, confidence = predict_fake_news(user_input)

        if prediction == 0:
            st.markdown("<h2 style='color: red;'>🟥 FAKE NEWS</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>🟩 REAL NEWS</h2>", unsafe_allow_html=True)

        st.write(f"🔎 Confidence: **{confidence:.2f}%**")
    else:
        st.warning("⚠️ Please enter some news text for analysis.")

# Footer / Info Section
st.markdown("---")
st.caption("This demo uses TinyBERT embeddings and Logistic Regression for educational purposes only. Predictions are based on headline/article content and may not be fully accurate.")

