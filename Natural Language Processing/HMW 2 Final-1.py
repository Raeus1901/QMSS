#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:33:52 2023

@author: jean
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct 25 17:28:26 2023
@author: jean
"""

import re
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import numpy as np

def clean(text):
    """Clean the text to keep only alphabets and spaces."""
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

def load_words_from_file(filename, file_encoding='utf-8'):
    """Load words from a given file."""
    with open(filename, 'r', encoding=file_encoding) as file:
        return set(word.strip().lower() for word in file.readlines())

# Load words using the respective encodings
pw = load_words_from_file("/Users/jean/Desktop/words/positive-words.txt", 'utf-8')
nw = load_words_from_file("/Users/jean/Desktop/words/negative-words.txt", 'ISO-8859-1')

def gen_senti(text, pw, nw):
    """Compute sentiment score based on presence of positive and negative words."""
    tokens = [word.lower() for word in word_tokenize(text)]
    
    positive_matches = [word for word in tokens if word in pw]
    negative_matches = [word for word in tokens if word in nw]

    pc = len(positive_matches)
    nc = len(negative_matches)

   # If at least one positive or negative word is present, compute S
    if pc > 0 and nc > 0:
        return (pc - nc) / (pc + nc)
    return None
  
"""Side note: One could change the above function to set a condition for sentiment score  
regardless if there is only positive or negative word by adjusting the or condition rather 
than and. The exercise clearly says to compute s ONLY IF there is any pc and nc 
in the text otherwise ignore. Because gloom is the only one, the s score is none.
"""
  
def rem_sw(text):
    """Remove stopwords from the text."""
    sw = nltk.corpus.stopwords.words('english')
    tokens = [word.lower() for word in word_tokenize(text)]
    return ' '.join(word for word in tokens if word not in sw)

# Display info about "gloom" and "prevail"
print("'gloom' is recognized in the negative dictionary" if "gloom" in nw else "'gloom' is not recognized in the negative dictionary")
print("'prevail' is recognized in the positive dictionary" if "prevail" in pw else "'prevail' is not recognized in the positive dictionary")
print("'darkest' is recognized in the negative dictionary" if "darkest" in nw else "'darkest' is not recognized in the positive dictionary")

# Test
text = "The darkest hour is among us in this time of gloom, however, we will prevail!"
cleaned_text = clean(text)
no_stopwords_text = rem_sw(cleaned_text)
sentiment_score = gen_senti(no_stopwords_text, pw, nw)


# Display processed data
print(f"Number of positive words: {len(pw)}")
print(f"Number of negative words: {len(nw)}")
print(f"Original text: {text}")
print(f"Cleaned text: {cleaned_text}")
print(f"Text without stopwords: {no_stopwords_text}")
print(f"Sentiment Score: S = {sentiment_score}")

# Load the dataframe and compute sentiment scores
the_data = pd.read_pickle("/Users/jean/Desktop/words/the_data.pk")
the_data['simple_senti'] = the_data['body'].apply(lambda x: gen_senti(x, pw, nw))
print(the_data.head())

# Using vaderSentiment
sentimentAnalyser = SentimentIntensityAnalyzer()
the_data['vader'] = the_data['body'].apply(lambda x: sentimentAnalyser.polarity_scores(x)['compound'])
print(the_data[['body', 'simple_senti', 'vader']].head())

# Handle NaN and infinite values
print("Number of NaN values in simple_senti:", the_data['simple_senti'].isna().sum())
the_data['simple_senti'].fillna(the_data['simple_senti'].mean(), inplace=True)
the_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Compute the statistics
print(f"Mean (Vader): {statistics.mean(the_data['vader'])}")
print(f"Mean (Simple Senti): {statistics.mean(the_data['simple_senti'])}")
print(f"Median (Vader): {statistics.median(the_data['vader'])}")
print(f"Median (Simple Senti): {statistics.median(the_data['simple_senti'])}")
print(f"Standard Deviation (Vader): {statistics.stdev(the_data['vader'])}")
print(f"Standard Deviation (Simple Senti): {statistics.stdev(the_data['simple_senti'])}")
