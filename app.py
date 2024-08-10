import pickle
import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def Transform_Text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorization.pkl','rb'))
model = pickle.load(open('modelBuilding.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Text preprocessing 
    transformed_sms = Transform_Text(input_sms)
    # 2. Text vectorization
    vector_input = tfidf.transform([transformed_sms])
    # 3. prediction
    result = model.predict(vector_input)[0]
    # 4. Displaying
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")