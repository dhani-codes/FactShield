# fact_shield.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---- Streamlit UI ----
st.title("🛡️ Fact Shield - Fake News Detector")
st.write("Enter some news text below to check if it’s True or Fake.")

# ---- Sample Dataset ----
data = {
    "text": [
        "The sky is blue today.",
        "Aliens have landed in India!",
        "Python is a programming language.",
        "Vaccines contain microchips."
    ],
    "label": [0, 1, 0, 1]  # 0 = True, 1 = Fake
}
df = pd.DataFrame(data)

# ---- Train ML Model ----
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = MultinomialNB()
model.fit(X, y)

# ---- User Input ----
user_input = st.text_area("Enter news text here:")

if st.button("Check News"):
    if user_input.strip() != "":
        # Predict using the trained model
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        if prediction == 0:
            st.success("✅ This news seems to be **True**.")
        else:
            st.error("⚠️ This news seems to be **Fake**.")
    else:
        st.warning("Please enter some text to check.")