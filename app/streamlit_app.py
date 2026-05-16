import streamlit as st
import pandas as pd
import joblib
import os
import sys

import matplotlib.pyplot as plt
import plotly.express as px


# Fix src path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from src.preprocess import clean_text

# Load trained model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Streamlit page config
st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    layout="wide"
)

# Main Title
st.title("Social Media Sentiment Analysis Dashboard")

st.markdown(
    """
    Analyze public sentiment from social media comments using
    Machine Learning and NLP.
    """
)

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.info(
    """
    This dashboard predicts:

    ✅ Positive Sentiment  
    ✅ Negative Sentiment  
    ✅ Neutral Sentiment  

    Technologies Used:
    - Python
    - NLP
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """
)

# User Input Section
st.subheader("Enter Social Media Comment")

user_input = st.text_area(
    "Type your comment here"
)

# Prediction Button
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Clean input
        cleaned = clean_text(user_input)

        # Vectorize
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)

        # Show prediction
        if prediction[0] == "positive":

            st.success(
                f"Predicted Sentiment: {prediction[0]} 😊"
            )

        elif prediction[0] == "negative":

            st.error(
                f"Predicted Sentiment: {prediction[0]} 😡"
            )

        else:

            st.warning(
                f"Predicted Sentiment: {prediction[0]} 😐"
            )

# Load Dataset
df = pd.read_csv(
    "data/raw/social_media_data.csv"
)

# Dataset Preview
st.subheader("Dataset Preview")

st.dataframe(df.head(10))

# Sentiment Distribution
st.subheader("Sentiment Distribution")

sentiment_counts = (
    df['sentiment']
    .value_counts()
    .reset_index()
)

sentiment_counts.columns = [
    'Sentiment',
    'Count'
]

# Bar Chart
bar_fig = px.bar(
    sentiment_counts,
    x='Sentiment',
    y='Count',
    color='Sentiment',
    title='Sentiment Distribution Bar Chart'
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)

# Pie Chart
pie_fig = px.pie(
    sentiment_counts,
    names='Sentiment',
    values='Count',
    title='Sentiment Distribution Pie Chart'
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)


# Sample Inputs
st.subheader("Sample Test Inputs")

st.code("This app is fantastic")

st.code("Worst customer support ever")

st.code("The update was released today")

# Footer
st.markdown("---")

st.markdown(
    "Developed using Machine Learning, NLP, and Streamlit 🚀"
)