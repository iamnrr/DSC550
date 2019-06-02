# -*- coding: utf-8 -*-

'''
Used in workflow1_prepdata.py
this performs the below tasks:
    splitting the data into train and test data and vectorize the data source
    The choice of vectorization is defined in the config file (CV or TFIDF)

'''

from utils import util

import pandas as pd
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# to suppress warnings
import warnings
warnings.filterwarnings('ignore')


def split(data):
    '''
    funciton to split the data, called from vectorizer function
    arguments: cleaned data from interim folder
    returns training data set and test data sets
    '''

    file = data
    data = pd.read_csv(file)    
    # making sure no nulls in features column
    data = data[pd.notnull(data['stemmedtxt'])]
    
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
    

def vectorizer(data, vectorizer, saveto):
    '''
    funciton to vectorize the data based on the parameter given in config file
    arguments: cleaned data, choice of vectorizer and path to save vectorized corpus
    returns vectorized training data set and testing data set (including features and targets)
    '''
    # reading configuration files
    config = util.get_config()
    m_features = int(config.get('MaxFeatures', 'm_features'))
    print('Max number of features selected : ', m_features)

    # calling split function and access the training and test data
    X_train, X_test, y_train, y_test = split(data)
    
    ext = '.pkl'  
    # applying the selected vectorizer to the split data    
    if vectorizer == "CV":
        # Creating count vectorizer objects
        catcv_vect = CountVectorizer( analyzer='word', 
                                     #stop_words='english',lowercase = True, 
                                        ngram_range=(2,3),       # ngrams - 2,3 
                                        max_features=m_features, # Had to restrict to 10000 features otherwise its running forever
                                        max_df = 0.5,  
                                        min_df = 3)         
        
        features_train = catcv_vect.fit_transform(X_train) 
        print(len(catcv_vect.get_feature_names()))            
        print(catcv_vect.fit_transform(X_train).shape)         
        features_test = catcv_vect.transform(X_test)
    
    elif vectorizer == "TFIDF":  
        # Creating TF-IDF vectorizer objects
        tfidf_vect = TfidfVectorizer(analyzer='word', 
                                     #stop_words='english',lowercase = True, 
                                     ngram_range=(2,3),
                                     max_features=m_features,
                                     max_df = 0.5,
                                     smooth_idf=True, 
                                     min_df=3)
        
        features_train = tfidf_vect.fit_transform(X_train) 
        print(len(tfidf_vect.get_feature_names()))          
        print(tfidf_vect.fit_transform(X_train).shape)         
        features_test = tfidf_vect.transform(X_test)
        
        
    # Confirming shapes of test and train transformations
    print('selected vectorizer is : ',vectorizer )
    print(features_train.shape)
    print(features_test.shape)
    
    filename = saveto+vectorizer+ext    
    print(filename)
    print('Vectorizer saved to location as : ', filename)
    
    # saving model to disk as specified in saveto
    joblib.dump(features_train, filename)
        
    return features_train, features_test, y_train, y_test