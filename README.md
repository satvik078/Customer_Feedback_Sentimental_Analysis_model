# Customer Feedback Sentiment Analysis AI Platform

## Overview

This project is an AI-powered Customer Feedback Sentiment Analysis Platform built using **BERT**, **Hugging Face Transformers**, and **Streamlit**. It analyzes customer reviews and classifies them as **Positive** or **Negative** with confidence scores.

The system supports multiple input sources including:
- Manual text input
- CSV file upload with smart column auto-detection
- API-based real-time review fetching with flexible JSON parsing

It is designed as a production-ready ML application with a premium dashboard interface and professional analytics.

---

## Features

### Sentiment Analysis using BERT
- Fine-tuned `bert-base-uncased` model
- Binary classification: Positive / Negative
- Confidence score prediction

### Smart CSV Handling
- Auto-detects text columns like `review`, `feedback`, `comment`, `content`
- Handles different CSV structures
- Manual column override support

### API Integration
- Accepts external API URLs
- Recursive JSON parsing for unknown API structures
- Optional manual field selection
- Compatible with SearchAPI and public APIs

### Interactive Dashboard
- Professional Streamlit UI
- Sentiment distribution charts
- Confidence distribution analytics
- Downloadable CSV results

### Hugging Face Integration
- Pretrained model hosted on Hugging Face
- No need to upload model files to GitHub

---

## Tech Stack

### Frontend
- Streamlit

### Machine Learning
- Hugging Face Transformers
- BERT (`bert-base-uncased`)
- PyTorch

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly

### API Handling
- Requests

### Deployment
- Render / Streamlit Community Cloud

---

## Project Structure

```text
Customer_Feedback_Sentiment_Analysis/
│
├── real_time.py
├── requirements.txt
├── runtime.txt
├── README.md
└── model
```

---

## Installation

```bash
git clone https://github.com/satvik078/Customer_Feedback_Sentimental_Analysis_model.git
cd Customer_Feedback_Sentimental_Analysis_model
pip install -r requirements.txt
streamlit run real_time.py
```

---

## Model Information

Hugging Face Model:
`Satvik078/Customer_Feedback_Sentimental_Analysis_model`

---

## Author

**Satvik Pandey**
GitHub: https://github.com/satvik078