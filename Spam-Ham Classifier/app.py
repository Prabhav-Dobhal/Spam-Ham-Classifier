import streamlit as st
import pickle
import nltk
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
def preprocessing(text):
    text=text.lower()
    #separate words
    text=nltk.word_tokenize(text)
    #remove special characters
    new = []
    for i in text:
        if i.isalnum():
            new.append(i)
    new = ' '.join(new)
    return new

st.title("Spam-Ham Classifier")
text = st.text_input("Enter text here")
if st.button('Predict'):
    newtext = preprocessing(text)
    vector_input = tfidf.transform([newtext])
    result = model.predict(vector_input)[0]
    if result == 1 :
        st.header("Spam")
    else:
        st.header("Ham")
