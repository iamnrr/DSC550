# -*- coding: utf-8 -*-
"""
@author: Nanduri Raghu Raman
Used in workflow1_prepdata.py
this performs the below tasks:
    cleaning the text file and encoding the categorical variable

"""


from utils import util
import pandas as pd


def preprocessing(file):
    '''
    function that uses utility functions to clean the text, lemmatization and stemming operations.
    Saves the data as the csv file in iterim directory (mentioned in config file)
    Arguments: the source file
    '''
    
    # reading category data set
    catdf = util.json_pd(file)
    # encode labels 
    labelencoder(catdf)    
    # Sampling huge data set and at the same time, keep the fraction of the category by using same samplng fraction across
    catdf0 = catdf[catdf.cat == 0 ].sample(frac=0.06, random_state= 1)
    catdf1 = catdf[catdf.cat == 1 ].sample(frac=0.06, random_state= 1)
    catdf2 = catdf[catdf.cat == 2 ].sample(frac=0.06, random_state= 1)
    catdf3 = catdf[catdf.cat == 3 ].sample(frac=0.06, random_state= 1)
    sampledata = pd.concat([catdf0, catdf1, catdf2, catdf3])
    
    #Applying text cleaning on text field to clean it up
    sampledata['clndtxt'] = sampledata['txt'].apply(util.textcleaning)
    # Apply lemmatization using util function
    sampledata['lemmedtxt'] = sampledata['clndtxt'].apply(util.lemmatize_text).apply(lambda x : " ".join(x))
    # Apply Stemming using util function
    sampledata['stemmedtxt'] = sampledata['lemmedtxt'].apply(util.stemmed_words) #.apply(lambda x : " ".join(x))
    # filtering rows that have null text after cleaning, lemmatization and stemming
    data = sampledata[pd.notnull(sampledata['stemmedtxt'])]
    
    #Get configuration file to read interim folder and file
    config = util.get_config()
    tempdatadir = config.get('Temp_datafolder', 'tempdatadir') 
    interim_data = tempdatadir+'interimdata.csv'
    data.to_csv(interim_data, sep = ',')    
    print('Interim csv loaded')    
    return


def labelencoder(dataframe):
    '''
    function to perform the label encoding
    Arguments: dataframe
    called from preprocessing function
    '''
    
    # Converting "cat" column to numerical encoding
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    
    sampledata = dataframe
    cat = sampledata['cat']
    # retaining original categorical field for further usage
    sampledata['cat_txt'] = sampledata['cat']
    sampledata['cat'] = encoder.fit_transform(cat)
    #check counts by different categorical groups
    print(sampledata.groupby('cat').count())
    
    return 
    
