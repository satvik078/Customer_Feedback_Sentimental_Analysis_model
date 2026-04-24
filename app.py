import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------
# CUSTOM CSS (PREMIUM UI)
# ---------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }

    .main {
        background: transparent;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }

    .subtitle {
        text-align: center;
        color: #dcdcdc;
        margin-bottom: 30px;
    }

    .card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0px 4px 30px rgba(0,0,0,0.2);
        margin-top: 20px;
    }

    .positive {
        color: #00ffcc;
        font-weight: bold;
        font-size: 24px;
    }

    .negative {
        color: #ff4b5c;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    sentiment = "Positive 😄" if predicted_class == 1 else "Negative 😞"
    
    return sentiment, confidence, predicted_class

# ---------------------------
# UI
# ---------------------------
st.markdown('<div class="title">💬 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer feedback using BERT 🤖</div>', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter your review:", height=150)

if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        sentiment, confidence, predicted_class = predict_sentiment(user_input)

        # Card UI
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if predicted_class == 1:
            st.markdown(f'<div class="positive">✅ {sentiment}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="negative">❌ {sentiment}</div>', unsafe_allow_html=True)

        st.write(f"**Confidence Score:** {confidence:.2f}")

        st.progress(int(confidence * 100))

        st.markdown('</div>', unsafe_allow_html=True)

        if predicted_class == 1:
            st.balloons()