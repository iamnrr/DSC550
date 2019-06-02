# -*- coding: utf-8 -*-
"""
@author: Nanduri Raghu Raman
"""

"""
Used in 
"""

# Importing required modules

import pandas as pd
import json
import re
import string
from datetime import datetime

from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import configparser


def get_config():
    config = configparser.ConfigParser()
    config.read(['config/config.cfg'])
    return config


def read_articles_jsonl(file_path):
    """
    function to read data from a file and return data
    """

    with open(file_path) as jsonfile:
        for line in jsonfile:
            data = json.loads(line)
    
    return data



def textcleaning(text):
    """
    Function to clean the text
    """
    # Converting to bytes and lower case
    text = str(text).lower()
    #removing \n
    text = re.sub(r'\\n', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'@\w+', '', text)
    # removing urls
    text = re.sub(r'http.?://[^\s]+[\s]?', ' ', text)
    # removing symbols and numbers
    text = re.sub('[^a-zA-Z\s]', '', text)
    # removing 3 letter words
    text = re.sub(r'\W*\b\w{1,3}\b', '', text)  #(r'(\b\w{1,3}\b', '', text)
    
    return text




def rem_stopwrds(mess):
    """
     Takes in a string of text, then performs the following:
    1. Remove all stopwords
    2. Returns a list of the cleaned text
    """
    return [word for word in mess if word.lower() not in stopwords.words('english')] 

def rem_punc(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Returns a list of the cleaned text
    """

    # Check characters to see if they are in punctuation
    #nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    #nopunc = ','.join(nopunc)
  
    # Now just remove any stopwords
    #return nopunc #[word for word in mess if word.lower() not in stopwords.words('english')]

    #return ','.join([char for char in mess if char not in string.punctuation])
    return ([char for char in mess if char not in string.punctuation])



def tokenize_text(text):
    """
    function to tokenize text data
    """
    for sent in text:
        row = word_tokenize(sent)
        return row
 

def lemmatize_text(text):
    
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()    
    return [lemmatizer.lemmatize(w) for w in tokens]



def stemmed_words(text):
    
    stemmer = SnowballStemmer('english')
    w_tokenizer = WhitespaceTokenizer()
    wrdslist = []
    for w in w_tokenizer.tokenize(text):
        lemwrd = stemmer.stem(w)
        wrdslist.append(lemwrd)
    return " ".join(wrdslist)



def split_line(text):

    # split the text
    words = text.split()
    # for each word in the line:
    for word in words:
        # print the word
        print(words)
        


def json_pd(file):
    
    """
    function to read data from a file and return data frame of the data
    """
    df = pd.DataFrame()
    #print("in json to df")
    with open(file) as jsonfile:            
        df = pd.DataFrame()
        df = pd.read_json(jsonfile, lines = 'True')
    return df

   
def create_pd(file_path):    
    """
    function to read data from all files and return data in data frame format
    """    
    allarticles_df = pd.DataFrame()
    
    for i in range(len(file_path)):
        file = file_path[i]
    
        with open(file) as jsonfile:            
            # load the dataset
            df = pd.DataFrame()
            df = pd.read_json(jsonfile, lines = 'True')
            allarticles_df = allarticles_df.append(df, ignore_index = True)        
    return allarticles_df
