# -*- coding: utf-8 -*-
'''
Used in workflow4_Modelling.py
this performs the below tasks:
    accesses the trained vocabulary list to apply for the new text for topic modelling 
'''


# Load libraries
from utils import util
from .task7_vectorizer_newtext import new_vectorizer

import pickle
from sklearn.externals import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import warnings
warnings.filterwarnings('ignore')

 

def loadvoacb(vectusing, modelusing, pickfrom, sample_text):
    '''
    function to load the trained voab
    arguments: 
      vectusing:choice of vectorizer
      modelusing: choice of model that needs to be applied
      pickfrom: pickled model to apply for new text for topic modelling
      sample_text: text for topic modelling
    returns: sample text given for topic modelling and the predicted label
    '''
    
    # assign models pickled filename to filename for accessing based on model parameter
    if modelusing == 'LRL1':        
        modelfile = 'LogisticRegressionL1.pkl'        
    elif modelusing == 'LRL2':
        modelfile = 'LogisticRegressionL2.pkl'    
    elif modelusing == 'NB':
        modelfile = 'NaiveBayes.pkl'             
    elif modelusing == 'RF':
        modelfile = 'RandomForest.pkl'

    #create filename using modelpath and modelfile
    filename = pickfrom+modelfile
    print(filename)
    # Load the model from the file 
    model_from_joblib = joblib.load(filename)  
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()
    
    data = config.get('Interim', 'Interim1')
    print('Pick cleaned data from : ', data)
   
    saveto = config.get('Modelpath', 'saveto') 
    print('Save model to : ', saveto)	
         
    # creates and pickles trained vocabulary 
    new_vectorizer(data, vectusing)    
    if vectusing == 'CV':
        filename = saveto+vectusing  
        trainedvectvoacb = pickle.load(open(filename , 'rb'))
        #reloading trained vocabulary
        loadedvect = CountVectorizer(vocabulary = trainedvectvoacb)        
    elif vectusing == 'TFIDF':
        filename = saveto+vectusing  
        trainedvectvoacb = pickle.load(open(filename , 'rb'))
        #reloading trained vocabulary
        loadedvect = TfidfVectorizer(vocabulary = trainedvectvoacb)
    
    # validating picled vocabulary
    loadedvect._validate_vocabulary()
    # transform the new text using the loaded vocabulary (for applying the trained model)
    newtestvect = loadedvect.transform(sample_text)

    # Use the loaded model to make predictions for sample test
    label_predictions = model_from_joblib.predict(newtestvect) 
    print(label_predictions, sample_text)    
    return (label_predictions, sample_text)

    
def modelling():
    '''
    funciton to perform the topic modelling or classification of the new test data
    and create a data frame of the modelled list
    '''
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()
    
    data = config.get('Interim', 'Interim1')
    print('Pick cleaned data from : ', data)
    
   
    vect = config.get('Vectorizer', 'Vect') 
    print('Selected Vectorizer is : ', vect)

    saveto = config.get('Modelpath', 'saveto') 
    print('Save model to : ', saveto)
    
    # read the model path location from config file
    pickfrom = config.get('Modelpath', 'pickfrom') 
    print('Save model to : ', pickfrom)
    
    # read the model from config file
    modelusing = config.get('TopicModelling', 'Model') 
    print('Use model  : ', modelusing)
    
    # read the required vectorization from config file
    vectusing = config.get('TopicModelling', 'Vect') 
    print('Use model  : ', vectusing)
    
    savemodelledto = config.get('TopicModelling', 'savemodelledto') 
    print('Save evaluation report to : ', savemodelledto)
    
    # read the required modelling text from config file
    sample  = config.get('TopicModelling', 'sample_text') 
    print('Sample Test is  : ', sample )  
    
    
    # clening the sample text using function in util module
    sample = util.textcleaning(sample)
    print(type(sample ))
    # converting string to list for it iterable
    sample_text = [sample]
    print(type(sample_text))
    print(sample_text)
    
    #creating a list of modelled topics and creating dataframe for further reporting
    modelleddata = []
    #sample_text = ["The Browns have a better QB situation than the 49ers.\n\nBringing up the Browns to absolve the 49ers of their poor decision making is incredibly lazy. Both teams can completely suck at QB evaluation"]
    modelleddata.append(loadvoacb(vectusing, modelusing, pickfrom, sample_text))
    # creates dataframe from the modelled data for generating report
    modelled_df = pd.DataFrame(modelleddata, columns = ['PredictedLabel', 'GivenText'])
    filename = savemodelledto+'topicmodelling.csv'
    # export modelled topic details to the csv
    modelled_df.to_csv(filename)
    return True
    
   

