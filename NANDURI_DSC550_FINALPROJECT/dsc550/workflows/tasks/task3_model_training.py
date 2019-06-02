# -*- coding: utf-8 -*-

'''
Used in workflow2_buildevaluate_model.py
this performs the below tasks:
    trains the models and saves them as pickles in the folder given in the configuration file
    
'''

# Importing local modules
from utils import util
from .task2_vectorizer import vectorizer
# Importing standard modules
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')



def modeltrain(saveto, model = None):
    '''
    function to train the data
    arguments - folder to save the model, 2nd parameter model is optional
    '''
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()   
    data = config.get('Interim', 'Interim1')
    vect = config.get('Vectorizer', 'Vect') 

     # training and test data -  features  & targets      
    X_train, X_test, y_train, y_test = vectorizer(data, vect, saveto) 
    if model == 'LRL1':
        # Creating instance for logistic regression with penalty L1
        model_clf = LogisticRegression(n_jobs=-1, 
                                   penalty='l1',
                                   multi_class='multinomial', 
                                   solver = 'saga',
                                   random_state=1) 
        modelnm = 'LogisticRegressionL1'        
    elif model == 'LRL2':
        # Creating instance for logistic regression with penalty L1
        model_clf = LogisticRegression(n_jobs=-1, 
                                   penalty='l2',
                                   multi_class='multinomial', 
                                   solver = 'saga',
                                   random_state=1) 
        modelnm = 'LogisticRegressionL2'    
    elif model == 'NB':
        # Create Multinomial NB classifier
        # added alpha = 1 for laplace smoothing for multi-classification
        model_clf = MultinomialNB(alpha = 1, class_prior=None, fit_prior=True) 
        modelnm = 'NaiveBayes'    
    elif model == 'RF':
        # Create random forest classifier object
        model_clf = RandomForestClassifier(n_jobs=-1, random_state=1)
        modelnm = 'RandomForest'
        
    # Applying classification model to train data set
    model_clf.fit(X_train, y_train)
    ext = '.pkl'
    # get date to append to the report
    time = datetime.now().strftime("%Y-%m-%d")
    #filename = saveto+modelnm+'_'+time+ext    
    filename = saveto+modelnm+ext 
    print('Model saved to location : ', filename)
    
    # saving model to disk as specified in saveto
    joblib.dump(model_clf, filename)    
    # predict the labels on trained dataset
    y_pred = model_clf.predict(X_test) 
    return y_test, y_pred
