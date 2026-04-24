import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import requests
import plotly.express as px

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Sentiment AI Platform", layout="wide")

# ---------------------------
# UI STYLE
# ---------------------------
st.markdown("""
<style>
.stApp { background: #0f172a; color: #e2e8f0; }
h1, h2, h3 { color: #f8fafc; }
.block { background: #1e293b; padding: 20px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "Satvik078/Customer_Feedback_Sentimental_Analysis_model"  # 🔥 CHANGE THIS
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return ("Positive" if pred == 1 else "Negative"), confidence

# ---------------------------
# CSV SMART DETECTION
# ---------------------------
def detect_text_column(df):
    possible_names = ["review", "text", "comment", "feedback", "content"]

    for col in df.columns:
        if col.lower() in possible_names:
            return col

    text_lengths = {}
    for col in df.columns:
        if df[col].dtype == "object":
            avg_len = df[col].astype(str).apply(len).mean()
            text_lengths[col] = avg_len

    if text_lengths:
        return max(text_lengths, key=text_lengths.get)

    return None

def load_csv(file):
    try:
        return pd.read_csv(file)
    except:
        return pd.read_csv(file, header=None)

# ---------------------------
# API SMART PARSER
# ---------------------------
def extract_text_from_json(data, min_length=20):
    texts = []

    if isinstance(data, dict):
        for value in data.values():
            texts.extend(extract_text_from_json(value))

    elif isinstance(data, list):
        for item in data:
            texts.extend(extract_text_from_json(item))

    elif isinstance(data, str):
        if len(data) > min_length:
            texts.append(data)

    return texts

def fetch_api_data(api_url):
    try:
        res = requests.get(api_url)
        data = res.json()
        texts = extract_text_from_json(data)
        return texts[:100]
    except:
        return []

def fetch_with_field(api_url, field):
    try:
        res = requests.get(api_url)
        data = res.json()

        if isinstance(data, dict) and field in data:
            data = data[field]

        texts = []
        if isinstance(data, list):
            for item in data:
                if field in item:
                    texts.append(str(item[field]))

        return texts[:100]
    except:
        return []

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Text", "CSV", "API"])

# ---------------------------
# HEADER
# ---------------------------
st.title("💬 Sentiment Analysis AI Platform")
st.caption("Smart system with CSV + API auto detection")

# ---------------------------
# TEXT MODE
# ---------------------------
if mode == "Text":
    text = st.text_area("Enter review")

    if st.button("Analyze"):
        if text.strip():
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict(text)

            col1, col2 = st.columns(2)
            col1.metric("Sentiment", sentiment)
            col2.metric("Confidence", f"{confidence:.2f}")
            st.progress(int(confidence * 100))

# ---------------------------
# CSV MODE
# ---------------------------
elif mode == "CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = load_csv(file)
        st.dataframe(df.head())

        auto_col = detect_text_column(df)

        if auto_col:
            st.success(f"Auto detected: {auto_col}")
        else:
            st.warning("Could not detect column")

        selected_col = st.selectbox("Select column", df.columns)
        col_to_use = auto_col if auto_col else selected_col

        if st.button("Analyze CSV"):
            df = df.dropna()
            texts = df[col_to_use].astype(str).tolist()[:200]

            sentiments = []
            confidences = []

            with st.spinner("Processing..."):
                for text in texts:
                    s, c = predict(text)
                    sentiments.append(s)
                    confidences.append(c)

            df_result = pd.DataFrame({
                "Text": texts,
                "Sentiment": sentiments,
                "Confidence": confidences
            })

            st.dataframe(df_result)

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(px.pie(df_result, names="Sentiment"))

            with col2:
                st.plotly_chart(px.histogram(df_result, x="Confidence"))

# ---------------------------
# API MODE
# ---------------------------
elif mode == "API":
    api_url = st.text_input("Enter API URL")
    field = st.text_input("Optional field (leave blank for auto-detect)")

    if st.button("Fetch & Analyze"):
        with st.spinner("Fetching data..."):
            if field:
                texts = fetch_with_field(api_url, field)
            else:
                texts = fetch_api_data(api_url)

        if not texts:
            st.error("No usable text found")
        else:
            sentiments = []
            confidences = []

            for text in texts:
                s, c = predict(text)
                sentiments.append(s)
                confidences.append(c)

            df_api = pd.DataFrame({
                "Text": texts,
                "Sentiment": sentiments,
                "Confidence": confidences
            })

            st.dataframe(df_api)

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(px.pie(df_api, names="Sentiment"))

            with col2:
                st.plotly_chart(px.bar(df_api["Sentiment"].value_counts()))

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("🚀 BERT + Hugging Face + Smart Data Handling")