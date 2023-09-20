from flask import Flask, request, jsonify
import joblib
import string
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the stacking model
stacking_model = joblib.load('stacking_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a set of stopwords
stopwordlist = [
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is',
    'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of',
    'on', 'once', 'only', 'or', 'other', ' our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she',
    "shes", 'should', "shouldve", 'so', 'some', 'such', 't', 'than', 'that', "thatll", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
    'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre", "youve", 'your', 'yours', 'yourself',
    'yourselves'
]

STOPWORDS = set(stopwordlist)

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

english_punctuations = string.punctuation

def cleaning_punctuations(text):
    translator = str.maketrans('', '', english_punctuations)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_URLs(data):
    return re.sub(r'((www.[^s]+)|(https?://[^s]+))', ' ', data)

def preprocess(text):
    text = text.lower()
    text = cleaning_stopwords(text)
    text = cleaning_punctuations(text)
    text = cleaning_repeating_char(text)
    text = cleaning_URLs(text)
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    return " ".join(text)

@app.route('/')
def index():
    return "Welcome to the Bail Prediction API"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            user_input = request.form['user_input']
        else:  # Handle GET request (if needed)
            user_input = request.args.get('user_input', '')  # Adjust the argument name as needed

        preprocessed_input = preprocess(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input])
        confidence_score = stacking_model.predict_proba(vectorized_input)[:, 1].item()
        return jsonify({'confidence_score': 10*confidence_score})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)