# -*- coding: utf-8 -*-

'''
Used in task8_topic_modelling_vect.py
this performs the below tasks:
    creates the vocabulary to apply models to the new text for topic modelling 
'''

# Import the moduls
import pandas as pd
import pickle
from utils import util

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def split(data):
    '''
    function to the split the data
    arguments: source file
    returns the features and targets of the trained and test data
    '''
    
    file = data
    data = pd.read_csv(file)    
    print(len(data))
    # making sure no nulls in features column
    data = data[pd.notnull(data['stemmedtxt'])]
    print(len(data))
    
    #split data into train and test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data['stemmedtxt'],
                                                                        data['cat'], 
                                                                        test_size = 0.25, 
                                                                        shuffle=True )
     # Checking lenghts of test and test
    print("Number of observations in train", len(X_train))    
    print("Number of observations in train", len(X_test))
    
    # return training data set and test data sets
    return X_train, X_test, y_train, y_test
    

def new_vectorizer(data, vectorizer):
    '''
    function to create the vocabulary using the choice of vectorizer
    arguments: source data and choice of vectorizer
    '''
    # split the data
    X_train, X_test, y_train, y_test = split(data)
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()
    
    # read urls from either from config file
    saveto = config.get('Modelpath', 'saveto') 
    print('Save model to : ', saveto)
    
    m_features = int(config.get('MaxFeatures', 'm_features'))
    print('Max number of features selected : ', m_features)
	
    filename = saveto+vectorizer    
        
    # saving model to disk as specified in saveto    
    if vectorizer == "CV":
        # Creating count vectorizer objects
        catcv_vect = CountVectorizer( analyzer='word', 
                                        stop_words = 'english',  # removes english stop words
                                        ngram_range=(2,3),       # ngrams - 2,3 
                                        max_features=m_features, # Had to restrict to 10000 features otherwise its running forever
                                        lowercase = True,  
                                        max_df = 0.5,  
                                        min_df = 3)         
        
        catcv_vect.fit(X_train) 
        pickle.dump(catcv_vect.vocabulary_, open(filename, 'wb'))
        print('Vectorizer saved to location as : ', filename)
        return
    
    elif vectorizer == "TFIDF":  
        # Creating TF-IDF vectorizer objects
        tfidf_vect = TfidfVectorizer(analyzer='word', 
                                     stop_words='english',
                                     ngram_range=(2,3),
                                     max_features=m_features,
                                     lowercase = True,  
                                     max_df = 0.5,
                                     smooth_idf=True, 
                                     min_df=3)
        
        tfidf_vect.fit(X_train)
        pickle.dump(tfidf_vect.vocabulary_, open(filename, 'wb'))
        print('Vectorizer saved to location as : ', filename)
        return

