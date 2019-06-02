# -*- coding: utf-8 -*-

'''
Reads parameerts required from configuration file
Execution:    
python wf_endtoend_model.py
'''

from utils import util
from workflows import workflow1_prepdata
from workflows import workflow2_buildevaluate_models
from workflows import workflow3_generatereport
from workflows import workflow4_Modelling


def endtoend_pipeline():
    '''
    function to perform end to end pipeline of 
        preprocessing
        vectorizing
        training models
        evaluating models
        generate evaluation metrics and report
        topic modelling and report on it
    '''

    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()    
        
    data = config.get('Interim', 'Interim1')
    print('Pick cleaned data from : ', data)
    
    vect = config.get('Vectorizer', 'Vect') 
    print('Selected Vectorizer is : ', vect)
    
    # read modelpath where the vectorized and fitted vector needs to be saved from config file
    saveto = config.get('Modelpath', 'saveto') 
    print('Save model to : ', saveto)
    
    workflow1_prepdata.initiatemodel_wf(data, vect, saveto)
    print('Workflow 1 done')
    
    results = workflow2_buildevaluate_models.train_wf()
    print('Workflow 2 done')
    
    workflow3_generatereport.gen_report(results)
    print('Workflow 3 done and report generated')
    
    workflow4_Modelling.topicmodelling_wf()
    print('Workflow 4 executed to generate topic modelling report')
    
    return True

def main():
    
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()        
    Rptfolder = config.get('Reportfolder', 'Rptfolder')
    endtoend_pipeline()
    print('Entire end to end pipeline executed :')
    print('Check the below directory for the reports generated\n')
    print('    - ', Rptfolder)    
    return True

if __name__ == '__main__':
    
    main()

    
    




