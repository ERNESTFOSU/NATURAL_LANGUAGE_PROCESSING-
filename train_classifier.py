import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy 
import nltk
nltk.download(['punkt' , "wordnet"])
from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle 


def load_data(database_filepath):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df["message"].values
    Y = df.drop(["message", "id" , "original" , "genre"] , axis = 1)
    return X,Y, Y.keys()
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline ([
      ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
       ('clf',  MultiOutputClassifier(RandomForestClassifier()))
       #('clf',  MultiOutputClassifier( KNeighborsClassifier()).fit(X, Y))
    ])
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        #'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
        #'clf__estimator__criterion': ['gini', 'entropy'],
        #'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    #cv.fit(X_train,Y_train)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict (X_test)
    i = 0
    for column in category_names :
    
        print(classification_report(Y_test.values[: ,i ], Y_pred[: ,i]) ,column )
        i=  i+ 1


def save_model(model, model_filepath):
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()