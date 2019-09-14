# -*- coding: utf-8 -*-
'''
used in wf_endtoend_model.py
performs the below tasks:
    generates report based on all the resulting metrics after applying different models
'''

from .tasks.task5_consolidate_evaluation import consolidate_results
from .tasks.task6_generatereport import to_markdown
from utils import util


def gen_report(eval_results):
    '''
    function to generate reports based on the evaluation metrics
    '''
    # calling the function to consolidation results
    consolidate_results(eval_results)
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()
    evalfileloc = config.get('Processed_datafolder', 'processeddatadir')
    print('Evaluated results are saved at : ', evalfileloc)
    
    #get the folder where the evaluation report needs to be saved
    reportpath = config.get('Reportfolder', 'Rptfolder')
    print('Pick evaluated results from csv : ', reportpath)
    
    #calling function to generate markdown report
    to_markdown(evalfileloc, reportpath)
    
    return True
