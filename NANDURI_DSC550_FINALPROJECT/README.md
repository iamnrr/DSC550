Please read the notes on configuration file and modify the config file in 
..\NANDURI_DSC550_FINALPROJECT\dsc550\config\config.cfg

***Due to the size of the source file, the original file is not being included and less number of features were considered for building models. 


End to end pipeline:  Performs end to end pipeline operations of building machine learning models and using those models to perform topic modelling. 
File: wf_endtoend_model.py
Involves the below workflows
Workflows
workflow1_prepdata: Workflow to prepare data and vectorize the data. Involves the below tasks defined in the following files:
	task1_preprocessing
task2_vectorizer	
workflow2_buildevaluate_models: Workflow to train and evaluate the model.  Involves the below tasks defined in the following files:
task3_model_training
task4_model_evaluation
workflow3_generatereport: Workflow to consolidate and generate the reports based on the trained models and their performance metrics. Involves the below tasks defined in the following files:
	task5_consolidate_evaluation
task6_generatereport
workflow4_Modelling: Workflow to use the trained models and perform topic modelling on new text. Involves the below tasks defined in the following files:
task7_vectorizer_newtext
task8_topic_modelling_vect
task9_generate_modellingrpt



This provides the guidance in using the code. The configuration file has the following settings.

directory of the source corpus
[Sourcefolder]
source = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\data\source\

directory of the saving the interim data
[Temp_datafolder]
tempdatadir = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\data\interim\

[Reportfolder]
# folder for saving the final reports
Rptfolder = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\reports\
#filepath and filename for saving the evaluation report in csv format
Evalfile = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\data\processed\eval_report.csv


directory for saving the processed file
[Processed_datafolder]
processeddatadir = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\data\processed\


choice of vectorizer to tokenize the text corpus (expected values "CV" for count vectorizer and "TFIDF" for TFIDF algorithm)
[Vectorizer] 
Vect = TFIDF 


THe below parameter lists all the algorithms that are passed as parameters to the model 
[Models]
Model1 = LRL1
Model2 = LRL2
Model3 = NB
Model4 = RF
Model5 = SCKMLP
Model6 = ANNCLF
Modellist = [LogisticRegressionL1, LogisticRegressionL2, MultinomialNB, RandomForest]


folders in which the pickled models need to saved and picked from
[Modelpath]
saveto = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\models\
pickfrom = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\models\


Maximum number of features for the vectorizers. For computational purposes made it as a configurable parameters
[MaxFeatures]
m_features = 10000

Directory structure from which the modelled topic needs to be picked, the model, vectorizer and the topic text corpus for modelling
[TopicModelling]
savemodelledto = C:\Users\nrrvlkp\Documents\M\550\NANDURI_DSC550_FINALPROJECT\reports\
Model = LRL1
Vect = CV
sample_text =  I think the game has been hyped very well.\n\nThe playable demo, the great plateau. There's a lot of footage online but it only reveals the surface of the game. There's still 98% of the game to see.\n\nThe trailer was excellent. Better than most movie trailers. It shows off the main characters and hints at a story but it's still very mysterious.\n\nLaunching with the Switch. This makes it doubly exciting for players like myself. I haven't owned a Nintendo since the GameCube (excluding handhelds) so to get my favorite series and a new console on the same day is going to give me a heart attack.



