import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import emoji

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
with open('classifier_lr_optimized.pickle', 'rb') as f:
    model = pickle.load(f)

with open('tfidfmodelUNIGRAM.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing functions from your notebook
def expand_contractions(text):
    dict_conts = {"ain't": "are not", "'s": " is", "aren't": "are not", "'re": " are", "'t": " not", " nt ": " not ", " u ":" you "}
    re_cont = re.compile('(%s)' % '|'.join(dict_conts.keys()))
    def replace(match):
        return dict_conts[match.group(0)]
    return re_cont.sub(replace, text)

def expand_hashtag(text):
    pattern = re.compile(r'#(\w+)')
    found_hashtags = re.finditer(pattern, text)
    for match in found_hashtags:
        spaced_hashtag = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', match.group(1))
        text = text.replace(match.group(), spaced_hashtag)
    return text

def pre_process_regex(text):
    tweet_tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['rt'])

    mention_pattern = re.compile(r'(?:rt)?@(\w+)')
    link_pattern = re.compile(r'https?://\S+|(www\.\S+)|(bit\.ly/\S+)|tinyurl\.\S+')

    text = expand_hashtag(text)
    text = text.lower()
    text = re.sub(mention_pattern, "", text)
    text = re.sub(link_pattern, "", text)
    text = expand_contractions(text)

    text = re.sub(r':|_|-', " ", text)
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s:]', " ", text)
    text = emoji.emojize(text)

    text = re.sub(r'\s+', " ", text)
    text = re.sub(r'\s\w\s', " ", text)
    text = re.sub(r'^\s+|\s+$', '', text)
    text = ' '.join([word for word in tweet_tokenizer.tokenize(text) if word not in stop_words])

    return text

# Prediction function
def predict_tweet(tweet):
    # Preprocess the tweet
    cleaned_tweet = pre_process_regex(tweet)
    # Vectorize the tweet
    tweet_vector = vectorizer.transform([cleaned_tweet])
    # Predict
    prediction = model.predict(tweet_vector)[0]
    # Get probability scores for all classes
    prob = model.predict_proba(tweet_vector)[0]
    # Get class names
    classes = model.classes_
    # Create a dictionary of class probabilities
    prob_dict = {cls: round(prob[i], 3) for i, cls in enumerate(classes)}
    return prediction, prob_dict

# Streamlit app
st.title("Cyberbully Tweet Classifier")
st.write("Enter a tweet to classify its cyberbullying type (e.g., religion, age, not_cyberbullying).")

# Input text box
user_input = st.text_area("Enter Tweet:", "")

# Predict button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        result, prob_dict = predict_tweet(user_input)
        st.success(f"Predicted Cyberbullying Type: **{result}**")
        st.write("**Class Probabilities:**")
        for cls, prob in prob_dict.items():
            st.write(f"{cls}: {prob:.3f}")
