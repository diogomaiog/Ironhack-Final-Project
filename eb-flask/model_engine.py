# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:19:09 2020

@author: Diogo Gonçalves
"""

import pickle
import pandas as pd
import regex as re
import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords

# list of valid movie genres
valid_genres = ['action','adventure','animation','biography','comedy','crime','documentary',
          'drama','family','fantasy','history','horror','music','musical','mystery',
          'news','romance','sci-fi','sport','thriller','war', 'western']

valid_ratings = ['Approved','G','NC-17','Not Rated','PG','PG-13','R','TV-14','TV-G','TV-MA','TV-PG',
                 'Unrated','X']

# load saved tables
valid_actors = pickle.load(open('valid_actors.pkl',"rb"))
valid_directors = pickle.load(open('valid_directors.pkl',"rb"))

# I need to import the naive bayes classifier
classifier = pickle.load(open('NBClassifier.pkl',"rb"))

# loading the previously created tables for ROI and Rating of people
actors_rating = pickle.load(open('actors_rating.pkl',"rb"))
directors_rating = pickle.load(open('directors_rating.pkl',"rb"))
actorsROI = pickle.load(open('actorsROI.pkl',"rb"))
directorsROI = pickle.load(open("directorsROI.pkl","rb"))

# loading the extreme gradient boost classifier
model = pickle.load(open('XGBClassifier.pkl',"rb"))

# Run Naive Bayes classifier on story

# Transform the string into a valid model input

# defining necessary funtions
def clean_up(s):
    """
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    # turn to lowercase   
    s = s.lower()
    #remove URLs
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    s = re.sub(regex,"",s)
    #replace special characters with spaces
    s = re.sub(r'([^a-zA-Z ]+?)', ' ', s)
    #removes leading and traling whitespaces
    s = s.strip()
    #removes multiple white spaces
    s = re.sub(' +', ' ', s)
    return s

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    #return s.split(" ")
    return nltk.word_tokenize(s)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper() # gets first letter of POS categorization
    tag_dict = {"J": wordnet.ADJ, 
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # get returns second argument if first key does not exist 

def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in l]

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    return [word for word in l if not word in stopwords.words()]

def process_text(text):
    '''
    Cleans-up, tokenizes, gets word net position, stems and lemmatizes and then removes the stopwords for a story text.
    Returns the list of resulting 'words'
    '''
    return remove_stopwords(stem_and_lemmatize(tokenize(clean_up(text))))

# I need to import the list of top5000 words
top5000 = pickle.load(open('top5000.pkl',"rb"))

# we need a function that finds the word features (our top most common words)
# in the text
def find_features(text):
    words = set(text)
    features = {}
    for w in top5000:
        features[w] = (w in words)
    return features

# make the prediction
def nlp_classifier(story):
    return classifier.classify(find_features(process_text('story')))

# computing starsIMDB, directorIMDB, starsROI, directorROI
def mean_ROI(names):
    '''
    takes a list of actors names and returns the average of average ROIs for the names provided
    '''
    values = []
    for name in names:
        values.append(float(actorsROI[actorsROI.primaryName == name]['ROI']))
    return sum(values)/len(values)

def dir_ROI(director_name):
    '''
    takes the name of a director and returns his/her average ROI
    '''
    return float(directorsROI[directorsROI.primaryName == director_name]['ROI'])

def mean_IMDB(names):
    '''
    takes a list of actors and returns the average of average IMDB ratings 
    for the names provided
    '''
    values = []
    for name in names:
        values.append(float(actors_rating[actors_rating.primaryName == name]['averageRating']))
    return sum(values)/len(values)

def dir_IMDB(director_name):
    '''
    takes the name of a director and returns his/her average IMDB score
    '''
    return float(directors_rating[directors_rating.primaryName == director_name]['averageRating'])

# put all inputs I can on a DF
def process_inputs(raw_inputs):
    '''
    takes a dictionary with user inputs, returns a dataframe ready to be
    used as input for the classifier model
    '''
    # creating empty df:
    inputs = pd.DataFrame(columns = ['startYear','runtimeMinutes','action','adventure','animation','biography','comedy','crime',
                                     'documentary','drama','family','fantasy','history','horror','music','musical','mystery',
                                     'news','romance','sci-fi','sport','thriller','war','western','result','Approved','G',
                                     'NC-17','Not Rated','PG','PG-13','R','TV-14','TV-G','TV-MA','TV-PG','Unrated','X',
                                     'starsIMDB','directorIMDB','starsROI','directorROI','NLPclass'], index =[0])

    # populating a dic that I will later add as a row on the inputs df
    inputs_dic = {}
    inputs_dic['startYear'] = raw_inputs['year']
    inputs_dic['runtimeMinutes'] = raw_inputs['runtimeMinutes']
    for g in valid_genres:
        if g in raw_inputs['genres']: 
            inputs_dic[g] = 1
        else:
            inputs_dic[g] = 0
    for r in valid_ratings:
        if r == raw_inputs['rating']: 
            inputs_dic[r] = 1
        else:
            inputs_dic[r] = 0
    raw_actors = set()
    raw_actors.add(raw_inputs['actor1'])
    raw_actors.add(raw_inputs['actor2'])
    raw_actors.add(raw_inputs['actor3'])
    inputs_dic['starsIMDB'] = mean_IMDB(list(raw_actors))
    inputs_dic['directorIMDB'] = dir_IMDB(raw_inputs['director'])
    inputs_dic['starsROI'] = mean_ROI(list(raw_actors))
    inputs_dic['directorROI'] = dir_ROI(raw_inputs['director'])
    inputs_dic['NLPclass'] = nlp_classifier(raw_inputs['story'])

    # creating a datframe and changing dtypes 
    inputs.loc[0] = pd.Series(inputs_dic)
    inputs.drop(columns='result',inplace=True)
    inputs = inputs.astype('float64')
    return inputs

def predict_outcome(inputs):
    '''
    takes inputs (a single row dataframe) and returns the predicted
    result flop, regular or blockbuster
    '''
    p = int(model.predict(inputs))
    if p == 0:
        return 'Flop'
    if p == 1:
        return 'Regular'
    if p == 2:
        return 'Blockbuster'
    
# MAIN
def run_prediction(data):
    raw_inputs = data
    return predict_outcome(process_inputs(raw_inputs))