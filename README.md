# Fact Shield 🛡️

## Tagline
"Spot the Fake, Trust the Truth."

## About
Fact Shield is a web app that detects fake news. Users can enter any news text and get a True/Fake prediction instantly.

## Inspiration
We created this project to help people identify fake news quickly and easily. With the rise of misinformation online, a simple tool to check news authenticity can make a real difference.

## What It Does
- Takes news text as input.
- Uses a simple machine learning model to predict if the news is True or Fake.
- Displays the result instantly in a user-friendly interface.

## How We Built It
- *Language:* Python  
- *Framework:* Streamlit  
- *Libraries/Technologies:* Pandas, scikit-learn, TF-IDF, Naive Bayes  
- *Development Environment:* VS Code  

## Challenges We Ran Into
- Handling text processing for fake news detection.
- Making sure the Streamlit app runs without errors locally.
- Understanding how to deploy the app online for public access.

## Accomplishments We’re Proud Of
- Created a fully functional web app that predicts fake news.
- Successfully built an ML model integrated into Streamlit.
- Learned how to deploy a Python app online.

## What We Learned
- Basics of text classification using TF-IDF and Naive Bayes.
- How to create interactive web apps with Streamlit.
- How to structure a project for deployment and GitHub sharing.

## How to Run Locally
1. Install dependencies:
```bash
pip install streamlit pandas scikit-learn

Run the app:
streamlit run fact_shield.py
Open the browser to http://localhost:8501 and test the app.

What's Next
Expand the dataset for more accurate predictions.
Improve the ML model and UI.
Deploy publicly so everyone can use it.

Try It Out
https://dhani-codes-factshield-fact-shield-oiiv2a.streamlit.app/
