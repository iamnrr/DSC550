# -*- coding: utf-8 -*-

'''
used in wf_endtoend_model.py
performs preprocessing and vectorization tasks
'''

from .tasks.task1_preprocessing import preprocessing
from .tasks.task2_vectorizer import vectorizer
from utils import util


def initiatemodel_wf(arg1, arg2, arg3):
    '''
    function to preprocess and vectorize as part of end to end pipeline
    takes 3 arguments called from end to end pipeline - read from config file    
    '''
        
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()    
    # read source folder from config file
    source = config.get('Sourcefolder', 'source') 
    preprocessing(source)
    print('Task1 done')  
    
    vectorizer(arg1, arg2, arg3)
    print('Task2 done')
    
    return True
 