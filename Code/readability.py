
# !pip install ReadabilityCalculator
# !pip install textstat


import textstat
from readcalc import readcalc
import nltk
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stops = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
import re


from textblob import TextBlob
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import *
import string


def compute_readability(df, col):
  '''computes the readability measures of text
  input:
  df = inpute dataframe 
  col = column of inpute dataframe for which the readability scores will bee calculated'''

  for ind, row in df.iterrows():
    calc = readcalc.ReadCalc(row[col])
    df.loc[ind,'smog_index'] = calc.get_smog_index()
    df.loc[ind,'flesch_reading_ease'] = calc.get_flesch_reading_ease()
    df.loc[ind,'flesch_kincaid_grade_level'] = calc.get_flesch_kincaid_grade_level()
    df.loc[ind,'coleman_liau_index'] = calc.get_coleman_liau_index()
    df.loc[ind,'gunning_fog_index'] = calc.get_gunning_fog_index()
    df.loc[ind,'ari_index'] = calc.get_ari_index()
    df.loc[ind,'lix_index'] = calc.get_lix_index()
    df.loc[ind,'dale_chall_score'] = calc.get_dale_chall_score()
    df.loc[ind,'dale_chall_known_fraction'] = calc.get_dale_chall_known_fraction()
  return df

def compute_syntactic(df, col):
  for ind, row in df.iterrows():
    df.loc[ind,'syllable_count'] = textstat.syllable_count(str(row[col]))
    #df.loc[ind,'lexicon_count'] = textstat.lexicon_count(str(row[col]), removepunct=True)
    df.loc[ind,'sentence_count'] = textstat.sentence_count(str(row[col]))
  return df

