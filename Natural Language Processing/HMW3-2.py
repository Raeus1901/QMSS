import sys
sys.path.append('/Users/jean/Desktop/')  # Adjust this path if necessary
from funcs.utils import *
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import collections
import nltk
from nltk.stem import PorterStemmer
import pickle

# Function to calculate Jaccard Distance
def jd(str_in_a, str_in_b):
    set_a = set(str_in_a.split())
    set_b = set(str_in_b.split())
    the_int = set_a.intersection(set_b)
    the_union = set_a.union(set_b)
    jd_t = len(the_int) / len(the_union)
    return jd_t

# Function to clean text
def clean_txt(str_in):
    tmp_clean_t = re.sub("[^A-Za-z']+", " ", str_in).lower().strip()
    return tmp_clean_t

# Function to read file
def file_reader(path_in):
    with open(path_in, "r", encoding="UTF-8") as f:
        tmp = f.read()
        tmp = clean_txt(tmp)
    return tmp

# Function to walk through files in a directory
def file_walker(data_path_in):
    t_data = pd.DataFrame()
    for root, dirs, files in os.walk(data_path_in, topdown=False):
        for name in files:
            path_t = os.path.join(root, name)
            try:
                txt_t = file_reader(path_t)
                if len(txt_t) > 0:
                    label_t = root.split("/")[-1]
                    tmp_data = pd.DataFrame({"body": txt_t, "label": label_t}, index=[0])
                    t_data = pd.concat([t_data, tmp_data], ignore_index=True)
            except Exception as e:
                print(f"Error processing {path_t}: {e}")
    return t_data

# Function to count words
def fun_wrd_cnt(pd_in, col_name_in):
    fun_word = dict()
    for topic in pd_in.label.unique():
        wrd_fun = pd_in[pd_in.label == topic]
        str_cat = wrd_fun[col_name_in].str.cat(sep=" ")
        fun_word[topic] = collections.Counter(str_cat.split())
    return fun_word

# Function to remove stopwords
def rem_sw(str_in):
    sw = nltk.corpus.stopwords.words('english')
    sent = [word for word in str_in.lower().split() if word not in sw]
    return ' '.join(sent)

# Function for stemming
def stem_fun(str_in):
    ps = PorterStemmer()
    sent = [ps.stem(word) for word in str_in.lower().split()]
    return ' '.join(sent)

# Function to count tokens
def token_count(str_in):
    return len(str_in.lower().split())

# Function to count unique tokens
def token_count_unique(str_in):
    return len(set(str_in.lower().split()))

# Function to read a pickle file
def read_pickle(path_in, file_name):
    with open(os.path.join(path_in, file_name + ".pk"), 'rb') as f:
        return pickle.load(f)

# Reading data from the pickle file
pickle_dir = "/Users/jean/Desktop/funcs/"  # Directory containing the pickle file
the_data = read_pickle(pickle_dir, "the_data")

# Applying text processing functions if they do not already exist in the data
if 'body_sw' not in the_data.columns:
    the_data["body_sw"] = the_data.body.apply(rem_sw)
    the_data["body_sw_stem"] = the_data.body_sw.apply(stem_fun)
    
def word_prob(token, column, df):
    # Initialize a dictionary to hold probabilities
    prob_dict = {'all': None, 'fishing': None, 'hiking': None, 
                 'machinelearning': None, 'mathematics': None}

    # Total token count across all documents
    total_tokens = df[column].str.split().explode().shape[0]
    print(f"Total tokens: {total_tokens}")

    # Count occurrences of the token across all documents
    token_count_all = df[column].str.contains(token, case=False, regex=False).sum()
    print(f"Token '{token}' count in all documents: {token_count_all}")

    # Calculate probability for 'all'
    if total_tokens > 0:
        prob_dict['all'] = token_count_all / total_tokens

    for topic in prob_dict.keys():
        if topic != 'all':
            # Subset the dataframe for the topic
            topic_df = df[df['label'] == topic]

            # Total tokens in the topic
            total_tokens_topic = topic_df[column].str.split().explode().shape[0]
            print(f"Total tokens in '{topic}': {total_tokens_topic}")

            # Count occurrences of the token in the topic
            token_count_topic = topic_df[column].str.contains(token, case=False, regex=False).sum()
            print(f"Token '{token}' count in '{topic}': {token_count_topic}")

            # Calculate probability
            if total_tokens_topic > 0 and token_count_topic > 0:
                prob_dict[topic] = token_count_topic / total_tokens_topic

    return prob_dict


"""
The question was formulated in a strange way so i put two type of answer: 
    
    The first one consists in experimenting the word_prob in changing its 
    parameters freely, showing one set of combination at a time
    
    The second one is a loop that will go over every combinations of tokens
    and column, then print all the possible combinations so 15
    
"""

"Method one"


token = "all"  # Adjust token as needed
column = "body_sw_stem"  # Ensure this column exists in 'the_data' DataFrame            
probabilities = word_prob(token, column, the_data)
print(probabilities)

"Method two => print all combinations"


tokens_list = ["mathematics", "fishing", "hiking", "machinelearning", "all"]  # Add all tokens you want to analyze
columns_list = ["body", "body_sw", "body_sw_stem"]  # Add all columns you want to analyze

# Iterate over every combination of token and column, calculate, and print probabilities
for token in tokens_list:
    for column in columns_list:
        probabilities = word_prob(token, column, the_data)
        print(f"Token: {token}, Column: {column}, Probabilities: {probabilities}")
            





# the_data["body_cnt"] = the_data.body.apply(token_count)
# the_data["body_cnt_u"] = the_data.body.apply(token_count_unique)
# the_data["body_sw"] = the_data.body.apply(rem_sw)
# the_data["body_sw_cnt"] = the_data.body_sw.apply(token_count)
# the_data["body_sw_cnt_u"] = the_data.body_sw.apply(token_count_unique)
# the_data["body_sw_stem"] = the_data.body_sw.apply(stem_fun)
# the_data["body_sw_stem_cnt"] = the_data.body_sw_stem.apply(token_count)
# the_data["body_sw_stem_u"] = the_data.body_sw_stem.apply(token_count_unique)



# # Testing the word_prob function
# token = "mathematics"  # Adjust token as needed
# column = "body"  # Ensure this column exists in 'the_data' DataFrame
# probabilities = word_prob(token, column, the_data)
# print(probabilities)

