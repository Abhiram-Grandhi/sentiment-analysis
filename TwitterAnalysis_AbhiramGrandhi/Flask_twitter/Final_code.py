# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:43:13 2021

@author: jeevankumar.p01
"""

# Data Analysis
import re
import nltk
#nltk.download('wordnet')
from tqdm import tqdm
import numpy as np
import contraction #contraction mapping
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

# Flask app
from flask import Flask,render_template, request
import pickle

app = Flask(__name__)

#loading the pcikle files
with open('model_pkl' , 'rb') as f:
    model = pickle.load(f)

with open('tfidfvect2.pkl' , 'rb') as f:
    joblib_vect = pickle.load(f)

#nltk 
tok = WordPunctTokenizer()
stop_words = set(stopwords.words('english'))

#preprocessing function
def preprocess_text(text):
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat,'', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    apostrophe_handled = re.sub("’", "'", lower_case)
    expanded = ' '.join([contraction.contraction_mapping[t] if t in contraction.contraction_mapping else t for t in apostrophe_handled.split(" ")])
    print(expanded)
    #sen= decontracted(lower_case)
    sentence = ' '.join(e.lower() for e in expanded.split() if e.lower() not in stop_words) #removing stop words
    words = tok.tokenize(sentence)
    return (" ".join(words)).strip()

#lemmatization 
def normalization(tweet_list):
    lemmatizer = WordNetLemmatizer()
    word_list=nltk.word_tokenize(tweet_list)
    return ' '.join([lemmatizer.lemmatize(w) for w in word_list])

#sentiment analysis page
@app.route('/sentiment', methods = ['GET','POST'])
def sentiment():
    
    TweetIn = request.form.get('TweetIn')
    print(TweetIn)
    text_pre= preprocess_text(TweetIn)
    print(text_pre)
    lemi_text= normalization(text_pre)
    print(lemi_text)
    
    if len(lemi_text)==0:
        lastout = None
    else:
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidfvect2.pkl", "rb")))
        tfidf = transformer.fit_transform(loaded_vec.transform(np.array([lemi_text])))
        lastout= model.predict(tfidf)
        print(lastout)
   
    if lastout == 0:
        label_out= 'Positive sentiment'
    elif lastout == 1:
        label_out= ' Negative sentiment'
    else:
        label_out= "No Useful text detected"
        
    return render_template('sentiment.html', name=TweetIn,Output=label_out, Extract_text=lemi_text)

#home page
@app.route('/')
def home():
    return render_template('index.html')

app.run(debug=False)