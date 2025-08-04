import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Load HuggingFace sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Set page layout
st.set_page_config(page_title="Advanced Sentiment Dashboard", layout="wide")

# App Header
st.markdown("## 🤖 Advanced Social Media Sentiment Dashboard (with HuggingFace)")
st.markdown("Analyze text using a powerful BERT-based sentiment model.")

st.divider()

# Text Input
st.markdown("### 🔍 Quick Sentiment Checker")
user_text = st.text_area("Type a review or post to analyze sentiment (HuggingFace):")

if user_text:
    result = sentiment_pipeline(user_text)[0]
    label = result['label']
    score = result['score']

    emoji = "🙂" if label == "POSITIVE" else "☹️"
    st.success(f"**Sentiment:** {emoji} {label} (Confidence: {score:.2f})")

st.divider()

# File Upload
st.markdown("### 📂 Upload CSV File to Analyze Multiple Posts")

uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("❌ The CSV must contain a 'text' column.")
    else:
        # Apply HuggingFace model to all rows
        def get_sentiment(text):
            result = sentiment_pipeline(str(text))[0]
            return result['label']

        df['Sentiment'] = df['text'].apply(get_sentiment)

        st.success("✅ Analysis complete using HuggingFace model!")
        st.dataframe(df[['text', 'Sentiment']])

        # Charts
        st.markdown("### 📊 Sentiment Distribution")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📉 Bar Chart")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='Sentiment', palette='Set2')
            ax1.set_title("Sentiment Distribution")
            st.pyplot(fig1)

        with col2:
            st.markdown("#### 🥧 Pie Chart")
            sentiment_counts = df['Sentiment'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
            ax2.set_title("Sentiment Share")
            st.pyplot(fig2)

        st.divider()
        st.markdown("✅ HuggingFace model upgrade complete. Submit with confidence! 💪")
