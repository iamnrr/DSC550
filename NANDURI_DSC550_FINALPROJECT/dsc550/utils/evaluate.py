# -*- coding: utf-8 -*-


from utils import util

# for measuring accuracy, precision, recall, f1 and auc scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')



def fn_multiclass_metrics(actual_label, predicted_label):
    """
    function that takes acutal labels and predicted labels and returns
    accuracy, auc, precision, recall and f1 scores
    average = 'weighted' for multi class classification
    """
    accuracy = accuracy_score(actual_label, predicted_label)
    precision = precision_score(actual_label, predicted_label, average = 'weighted')
    recall = recall_score(actual_label, predicted_label, average = 'weighted')
    f1 = f1_score(actual_label, predicted_label, average = 'weighted')

    return (accuracy, precision, recall, f1)



def print_metrics(actual_label, predicted_label):

        
    conf_matrix = confusion_matrix(actual_label, predicted_label)
    print("\n Confusion Matrix \n")
    print(conf_matrix)

    #Generate labelled performance metrics
    print("\n Classification Report \n")
    print(classification_report(actual_label, predicted_label))
    accuracy, precision, recall, f1 = fn_multiclass_metrics(actual_label, predicted_label)

    print("Accuracy : ", accuracy)
    print("Precision ", precision)
    print("Recall : ", recall)
    print("F1 Score ", f1)

    return (accuracy, precision, recall, f1)


def classifier_metrics(classifier, X_test, y_test):

    # predict test label probabilities
    label_predict_prod = classifier.predict(X_test)
    print("\n Predicated labels on test data : \n ")
    print(label_predict_prod)
    
    # predict test labels
    label_predict = classifier.predict_classes(X_test)
    print("\n Predicated labels : \n ")
    print(label_predict)
    
    conf_matrix = confusion_matrix(y_test, label_predict)
    print("\n Confusion Matrix \n")
    print(conf_matrix)

    #Generate labelled performance metrics
    print("\n Classification Report \n")
    print(classification_report(y_test, label_predict))
    accuracy, precision, recall, f1 = fn_multiclass_metrics(y_test, label_predict)

    print("Accuracy : ", accuracy)
    print("Precision ", precision)
    print("Recall : ", recall)
    print("F1 Score ", f1)


    return (accuracy, precision, recall, f1)


#
#def main():
#    
#    # Get the configuraton settings to read URLs and symbols
#    config = util.get_config()    
#    
#    pickfrom = config.get('Modelpath', 'pickfrom')
#    print('Pick fitted model from : ', pickfrom)
#    
#    # Load the model from the file
#    trained_model = joblib.load(pickfrom)
#    
#    
#    print_metrics(trained_model, )
#    
#    
#    
#if __name__ == "__main__":
#    main()
