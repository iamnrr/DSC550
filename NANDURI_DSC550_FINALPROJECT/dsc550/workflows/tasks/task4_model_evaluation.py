# -*- coding: utf-8 -*-
'''
Used in workflow2_buildevaluate_model.py
this performs the below tasks:
    evaluates the trained models
    prints and saves the result sets as list for all the model
    
'''

from utils import util
from utils import evaluate
from .task2_vectorizer import vectorizer
from .task3_model_training import modeltrain

import warnings
warnings.filterwarnings('ignore')


def evaluate_model():
    '''
    function to evaluate the model and save the result as list
    arguments: none
    returns the result set as list
    '''
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()    
    
    # List to store model evaluation results
    allmodels_evalresults = []
    
    pickfrom = config.get('Modelpath', 'pickfrom')
    print('Pick fitted model from : ', pickfrom)
    
    data = config.get('Interim', 'Interim1')
    print('Pick cleaned data from : ', data)    
   
    vect = config.get('Vectorizer', 'Vect') 
    print('Selected Vectorizer is : ', vect)
    
    saveto = config.get('Modelpath', 'saveto') 
    print('Save model to : ', saveto)
    
    # getting max features parameter from the config file
    m_features = int(config.get('MaxFeatures', 'm_features'))
    
    X_train, X_test, y_train, y_test = vectorizer(data, vect, saveto)    
       
    # Get list of models
    model1 = config.get('Models', 'Model1') 
    model2 = config.get('Models', 'Model2') 
    model3 = config.get('Models', 'Model3') 
    model4 = config.get('Models', 'Model4')
    # creating a list of the parameters
    Models_list = [model1, model2, model3, model4]
    print('Models being evaluated are : ', Models_list)    
    # looping through the models list for training and printing the evaluation metrics
    for i in range(len(Models_list)):        
        model = Models_list[i]
        print('Selected model is : ', model)    
        y_test, y_pred = modeltrain(saveto, model=model)    
        accuracy, precision, recall, f1 = evaluate.print_metrics(y_test, y_pred)        
        #create a tuple of all parameters
        row = (model, vect, m_features, accuracy, precision, recall, f1)
        allmodels_evalresults.append(row)
       
    return allmodels_evalresults
    

    