# -*- coding: utf-8 -*-

'''
used in wf_endtoend_model.py
performs the below tasks:
    topic modelling based on the new text entered in the configuration file
    and generates report based on the modelling after applying model of choice (set in config file)
'''

from .tasks.task8_topic_modelling_vect import modelling
from .tasks.task9_generate_modellingrpt import gen_final_modellingreport
from utils import util


def topicmodelling_wf():
    '''
    function for modelling the topic
    '''
    # call the modelling function to perform topic modelling
    modelling()
    print('Task8 done')  
    # call the function to generate report of topic modelling
    gen_final_modellingreport()
    print('Task9 done')
    
    return True

