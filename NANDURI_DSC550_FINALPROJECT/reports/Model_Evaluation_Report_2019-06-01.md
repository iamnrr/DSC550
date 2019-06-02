
## Raghu Raman Nanduri - Final Project 510  

*** Evalutaion Report for the trained models on test data *** 


|   Unnamed: 0 | Model   | Vectorizer   |   Max features |   Accuracy |   Precision |   Recall |       F1 |
|-------------:|:--------|:-------------|---------------:|-----------:|------------:|---------:|---------:|
|            0 | LRL1    | CV           |          20000 |   0.50662  |    0.553362 | 0.50662  | 0.453121 |
|            1 | LRL2    | CV           |          20000 |   0.507257 |    0.542636 | 0.507257 | 0.457774 |
|            2 | NB      | CV           |          20000 |   0.509575 |    0.513413 | 0.509575 | 0.483273 |
|            3 | RF      | CV           |          20000 |   0.459831 |    0.443696 | 0.459831 | 0.429848 |
|            0 | LRL1    | TFIDF        |          50000 |   0.514645 |    0.566246 | 0.514645 | 0.460893 |
|            1 | LRL2    | TFIDF        |          50000 |   0.525046 |    0.567625 | 0.525046 | 0.482521 |
|            2 | NB      | TFIDF        |          50000 |   0.528754 |    0.598511 | 0.528754 | 0.478287 |
|            3 | RF      | TFIDF        |          50000 |   0.477533 |    0.477336 | 0.477533 | 0.459372 |



The model with better accuracy is :
    Unnamed: 0             2
Model                 NB
Vectorizer         TFIDF
Max features       50000
Accuracy        0.528754
Precision       0.598511
Recall          0.528754
F1              0.478287
Name: 6, dtype: object