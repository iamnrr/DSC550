# -*- coding: utf-8 -*-

'''
Used in workflow3_generatereport.py
this performs the below tasks:
    consolidates all the results of the metrics from the trained model    
'''
import pandas as pd
from utils import util


def consolidate_results(eval_result):  
    '''
    function to save the metrics from all the trained models and save the results as 
    csv and place in the process folder
    '''

    # create data frame to hold results
    multiclass_modelperf_df = pd.DataFrame()
    multiclass_modelperf_df = pd.DataFrame(eval_result, columns = ['Model','Vectorizer', 'Max features','Accuracy', 'Precision', 'Recall', 'F1'])
    print(multiclass_modelperf_df)
        
    # Get the configuraton settings
    config = util.get_config() 
    # fetch the path for the result csv
    savecsvto = config.get('Processed_datafolder', 'processeddatadir') 
    print('Save evaluation report to : ', savecsvto)
    # fetch the choice of vectorizer to create a proper file name
    vect = config.get('Vectorizer', 'Vect') 
    print('Selected Vectorizer is : ', vect)    
    filename = savecsvto+'eval_report_'+vect+'.csv'
    # export evluation report to the csv
    multiclass_modelperf_df.to_csv(filename)
    print('Evaluation results exported')

    return True