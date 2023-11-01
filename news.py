# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the trained model and CountVectorizer
model = pickle.load(open('nlp_model.pkl', 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))


# Streamlit UI
st.title("Fake News Detection Web App")
st.write("This app detects whether a news article is fake or real.")

# Input text box for user to enter news article
user_input = st.text_area("Enter the news article:")

# Predict fake news when the user clicks the button
if st.button("Detect Fake News"):
    if user_input:
        # Vectorize the user input
        user_input_vectorized = vectorizer.transform([user_input])

        # Make prediction
        is_fake = model.predict(user_input_vectorized)[0]

        # Display the prediction
        if is_fake:
            st.error("This news article is likely fake.")
        else:
            st.success("This news article appears to be real.")
    else:
        st.warning("Please enter a news article for analysis.")

# Display statistics and insights about fake news
st.header("Fake News Statistics and Insights")
st.write("Here are some statistics and insights about fake news:")

# You can replace this with actual statistics
st.write("The issue of fake news is a complex and evolving challenge in the digital age. Recognizing and combating fake news is vital for maintaining an informed and responsible society. Efforts from media outlets, technology companies, fact-checkers, and individuals are crucial in addressing this issue and preserving the integrity of news and information.")


