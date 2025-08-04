# 💬 AI-Powered Social Media Sentiment Dashboard

This is a real-time dashboard that uses a **HuggingFace Transformers model** to detect sentiment (Positive or Negative) in social media text, product reviews, or any user-generated content.

## 🚀 Features

- 🔍 Real-time sentiment detection for single-line input
- 📂 Upload CSV with multiple reviews for batch sentiment analysis
- 🤖 Powered by `distilbert-base-uncased-finetuned-sst-2-english` model from HuggingFace
- 📊 Visualizations using bar chart and pie chart
- ✅ Built with Streamlit (no HTML/JS required)

## 🧠 Tech Stack

- **Frontend/UI**: Streamlit
- **AI Model**: HuggingFace Transformers + DistilBERT
- **Backend Engine**: PyTorch
- **Data Handling**: Pandas
- **Charts**: Seaborn, Matplotlib

## 🧪 Model Used

- **Name**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Source**: HuggingFace Transformers
- **Task**: Sentiment Classification
- **Outputs**: Label (`POSITIVE` / `NEGATIVE`) + Confidence Score

## 🖥️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
