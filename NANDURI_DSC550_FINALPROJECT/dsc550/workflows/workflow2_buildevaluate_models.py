# -*- coding: utf-8 -*-

'''
used in wf_endtoend_model.py
performs the below tasks:
    train and evaluates the models
'''

# import modules
from .tasks.task4_model_evaluation import evaluate_model



def train_wf():
    '''
    perofmrs training and evaluation of the models and returs the evaluation results
    '''
    
    # List to store model evaluation results
    allmodels_evalresults = []      
    allmodels_evalresults = evaluate_model()
    print(allmodels_evalresults)
    print('Model Evaluated')
    return allmodels_evalresults