# -*- coding: utf-8 -*-

'''
Used in workflow3_generatereport.py
this performs the below tasks:
    reads the evaluation metrics saved in csv and generates reports   
'''

# Import required modules

from utils import util
import pandas as pd
import glob
import tabulate
from datetime import datetime


def read_evalreport(evalfileloc):
    '''
    function to read the csv with metrics
    arguments: path to the location of the csv with metrics
    returns the dataframe
    called from to_markdown function
    '''
    
    # Reading both the files in the diectory ( CV & TFIDF)
    files = glob.glob(evalfileloc+"/*.csv")
    print(files)
    
    all_results = []
    
    for file in files:
        data_df = pd.read_csv(file)    
        print(len(data_df))
        all_results.append(data_df)
    
    all_results_df = pd.concat(all_results, axis = 0, ignore_index = True)
    all_results_df = all_results_df.reset_index(drop = True)
    return all_results_df
    
    

def to_markdown(evalfileloc, reportpath):
    '''
    Function that takes path of a csv file and generates markdown report in the given path.
    Args:
        evalfileloc: location all the evaluation reports from various models
        reportpath: location where the markdown report should be saved
    Result:
        prints the results of the evaluation report to markdown report
    '''
    # time to append to the generated report
    time = datetime.now().strftime("%Y-%m-%d")
    reportfile = reportpath+'Model_Evaluation_Report_'+time+'.md'
    # data frame from the result metrics
    df = read_evalreport(evalfileloc)
    # to drop index from dataframe and evaluation report
    df = df.reset_index(drop = True) 
    # get the row with max accuracy
    print("Model with better Accuracy is : \n")
    row = df.loc[df['Accuracy'].idxmax()]
    print(row)
    # Write into markdown report
    with open(reportfile, 'w') as ofile:
        ofile.write('\n## Raghu Raman Nanduri - Final Project 510  \n\n')
        ofile.write('*** Evalutaion Report for the trained models on test data *** \n')
        ofile.write('\n\n')
        #write the data frame results to the markdown report
        ofile.write(tabulate.tabulate(df.values, df.columns, tablefmt = "pipe", showindex = False ))
        ofile.write('\n\n\n\n')
        ofile.write('The model with better accuracy is :\n')
        ofile.write('    %s' %(row))
    return True
