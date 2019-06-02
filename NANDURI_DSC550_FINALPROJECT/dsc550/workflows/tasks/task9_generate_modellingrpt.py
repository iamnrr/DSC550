# -*- coding: utf-8 -*-
'''
Used in workflow4_Modelling.py
this performs the below tasks:
    generates the report on the modelled topic
'''

# Import required modules

from utils import util
import pandas as pd
import tabulate
from datetime import datetime
import numpy as np

    

def to_markdown(evalfileloc, reportpath):
    """
    Function that takes path of a csv file and generates markdown report in the given path.
    Args:
        evalfileloc: location all the evaluation reports from various models
        reportpath: location where the markdown report should be saved
    Result:
        prints the results of the evaluation report to markdown report
    """
    # get date to append to the report
    time = datetime.now().strftime("%Y-%m-%d")
    reportfile = reportpath+'Topic_Model_Evaluation_Report_'+time+'.md'
    file = reportpath+'topicmodelling.csv'    
    data_df = pd.read_csv(file) 
    
    # dictionary for mapping the encoding labels to text categorical labels
    label_map = {'[0]': 'news', '[1]': 'science_and_technology', '[2]': 'sports', '[3]':'video_games'}
    data_df['PredictedLabelText'] = data_df['PredictedLabel'].map(label_map)
    # printing the topic modelling data frame
    print(data_df)
    
    # write the topic modelling results
    with open(reportfile, 'w') as ofile:
        ofile.write('\n## Raghu Raman Nanduri - Final Project 510  \n\n')
        ofile.write('*** Topic Modelling Report for the trained models on test data *** \n')
        ofile.write('\n\n')
        #write the data frame results to the markdown report
        for i in range(len(data_df)):
            ofile.write('Given Topic is :\n' )
            ofile.write('   - %s' % (data_df['GivenText'][i]))
            ofile.write('\n\n')
            ofile.write('Topic is classified as :\n' )
            ofile.write('   - %s' % (data_df['PredictedLabelText'][i]))               
        ofile.write('\n\n')
        # printing the whole date in tabular form        
        ofile.write(tabulate.tabulate(data_df.values, data_df.columns, tablefmt = "pipe", showindex = False ))
            

def gen_final_modellingreport():
    '''
    function that create the topic modelling report
    
    '''
    # Get the configuraton settings to read URLs and symbols
    config = util.get_config()
    evalfileloc = config.get('Processed_datafolder', 'processeddatadir')
    print('Evaluated results are saved at : ', evalfileloc)
    
    #get the folder where the evaluation report needs to be saved
    reportpath = config.get('Reportfolder', 'Rptfolder')
    print('Pick evaluated results from csv : ', reportpath)
    
    #calling function to generate markdown report
    to_markdown(evalfileloc, reportpath)
    return
    