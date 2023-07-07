import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError

# Load the pre-trained logistic regression model
model = pickle.load(open('logistic_regression.sav', 'rb'))
# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing functions
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
try:
    vectorizer = pickle.load(open('count_vectorizer_var.pkl', 'rb'))
except (FileNotFoundError, NotFittedError):
    pass

def vectorize_text(text):
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

# Streamlit app
def main():
    st.title("Fake News Detection")

    # Get user input
    user_input = st.text_input("Enter a news headline or article text:")

    if st.button("Check"):
        # Preprocess user input
        preprocessed_input = preprocess_text(user_input)

        # Vectorize preprocessed input
        vectorized_input = vectorize_text(preprocessed_input)

        # Make prediction using the pre-trained model
        prediction = model.predict(vectorized_input)

        # Display the prediction
        if prediction[0] == 0:
            st.error("Fake News Detected!")
        else:
            st.success("Real News Detected!")

# Run the app
if __name__ == '__main__':
    main()