import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.metrics import roc
# from sklearn.model_selection import traintestsplit
# from sklearn.preprocessing import 
import json
import warnings
# import library as lib
import nltk
from nltk.corpus import stopwords
# nltk.download('averaged_perceptron_tagger')
import string
from nltk.stem.wordnet import WordNetLemmatizer, wordnet
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
from nltk.collocations import *
from nltk import FreqDist
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

#### CLEANING FUNCTIONS


def full_clean(raw_data_json_file):
    
    raw_data = pd.DataFrame(pd.read_json(raw_data_json_file))

    # use json lib to parse first level of json file
    raw_data = pd.io.json.json_normalize(raw_data.data)
    data=raw_data.copy()
    # creating df that contains the questions, answers and context only
    qac = data.paragraphs.apply(pd.io.json.json_normalize)
    
    #create empty dataframe with extra space
    df = pd.DataFrame(data ={ 'context':[x for x in range(150000)], 
                             'question': [x for x in range(150000)], 
                             'answers': [x for x in range(150000)], 
                             'plausible_answers': [x for x in range(150000)], 
                             'is_impossible': [x for x in range(150000)]})
    for c in df.columns:
        df[c] = 0

    k=0
    n=0
    for j in qac:
        for i in j['qas']:
            cont = str(j['context'][n])
            for m in i:
                if m['is_impossible']==True:
                    df['is_impossible'][k] = 1 
                    df['question'][k] = m['question']
                    df['plausible_answers'][k] = m['plausible_answers'][0]['text']
                    df['answers'][k] = ''
                    df['context'][k] = cont
                else:
                    df['question'][k] = m['question']
                    df['plausible_answers'][k] = ''
                    df['answers'][k] = m['answers'][0]['text']
                    df['context'][k] = cont
                k+=1

            n+=1
        n=0
        
    df = df.loc[df.context!=0]
        
    return df
        
#### STEMMING, LEMMING and POS-TAGGING RELATED FUNCTIONS
    
    
def preprocessing(df, columns_list=['context', 'question'], 
                  lower_all=False, rm_stopwords=True, 
                  lemm=False, stem=False, pos_tagging=True, 
                  punct_to_keep=[], suffix='lemma_&_pos'):
    """ Here we take out unnecessary special characters; tokenize sentences; 
    tokenize within the sentences; remove stop words; add POS tags and, if specified, carry out 
    lemmatization and, where that is not possible, stemming. 
    Lemming and stemming have been left as default: False because we would most likely 
    want to keep the morphological structure of the question and context words intact, 
    given that we are producing predictions that are subsegments of the text span. 
    """
    if stem:
        porter = PorterStemmer()
    if lemm:
        lemmy = WordNetLemmatizer()
    
    
    stop_words = set(stopwords.words('english'))
    if len(punct_to_keep)==0: 
        string_mod_punctuation = string.punctuation
    else:
        string_mod_punctuation=string.punctuation
        for x in string_mod_punctuation:
            if x in punct_to_keep:
                string_mod_punctuation = string_mod_punctuation.replace( x , "")

    new_df = df.copy()
    new_df.reset_index(drop=True, inplace=True)
    
    for col in columns_list:
        
        new_df[col].astype('str')
        new_df[col] = new_df[col].apply(lambda x: x.replace('\n', ' '))

        new_df['{col}_{suffix}'.format(col=col, suffix=suffix)] = 0
        for i in range(len(new_df[col])):
            if len(new_df[col][i])>0:
                final_words=[]
                sentences = nltk.sent_tokenize(new_df[col][i])
                for sentence in sentences: 
        
                    input_words = sentence.translate(str.maketrans('','', string_mod_punctuation)).split(' ')
                    if rm_stopwords:
                        words = [word for word in input_words if not word in stop_words]
                    else:
                        words = [word for word in input_words]
                    words = [word for word in words if word]
                    
                    if lower_all:
                        words = [word.lower() for word in words]
                    if pos_tagging:
                        tagged_words=nltk.pos_tag(words)
                        for w in tagged_words:
                            if lemm:
                                morph_fn = get_morph(w[1])
                                if morph_fn!='':
                                    new_w = (lemmy.lemmatize(w[0], morph_fn), w[1])
                                else:
                                    new_w = w
                            elif stem:
                                new_w = (porter.stem(w[0]), w[1])
                            else:
                                new_w=w
                            final_words.append(new_w)
                    else:
                        for w in words:
                            final_words.append(w)
                new_df['{col}_{suffix}'.format(col=col, suffix=suffix)][i] = final_words
            else:
                new_df['{col}_{suffix}'.format(col=col, suffix=suffix)][i] = ''
    return new_df



def get_morph(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('S'):
        return wordnet.ADJ
    elif pos_tag.startswith('A'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    
def extract(lst_tup):
    output=''
    for i in lst_tup:
        output+=i[0]+' '
    return output


def prep_for_lstm(data, columns=['context', 'question', 'answers'],
                 ):
    
    pass
#### CORPUS STATISTICS RELATED FUNCTIONS
    
    
def conc_str(lst):
    """Adds together the words from a list of tuples
    of preprocessed words and their POS tag"""
    conc_str=''
    for i in lst:
        conc_str+=i[0]
        conc_str+=' '
    return conc_str


def get_freq_dist(df, df_cols=['context_lemma_pos', 'question_lemma_pos'], drop_SET_col=True, topn=50):
    """returns a list of tuples containing a string and its frequency from within the 
    total corpus. This function ASSUMES the values passed into it contain a TUPLE, 
    where the 0th value is the token you're after. 
    df - dataframe
    df_cols - list; which columns are to be used as the corpus
    drop_SET_col - boolean; the fn creates a column with just the tokens for each row; 
    if true, removes the newly created column after it's been used;
    topn - how many of the most frequent in the corpus you'd like to see returned
    """
    
    total_set_tokens=[]
    
    for col in df_cols:
        df['{col}_SET_TOKENS'.format(col=col)] = df[col].apply(conc_str)

        tokens_set_list = set(df['{col}_SET_TOKENS'.format(col=col)])

        
        for x in tokens_set_list:
            tokenized = word_tokenize(x)
            for y in tokenized:
                total_set_tokens.append(y)
        if drop_SET_col:
            df.drop(columns=['{col}_SET_TOKENS'.format(col=col)], inplace=True)

    test_freqdist = FreqDist(total_set_tokens)
    return test_freqdist.most_common(topn)

def get_tf_df(token_list):
    """computes the term frequency for a sentence of tokens"""
    tf_df = pd.DataFrame(index=set(token_list), columns=['tf'])
    for t in set(token_list):
        t_count=0
        for w in token_list:
            if t==w:                
                t_count+=1
        tf_df['tf'][t] = t_count/len(token_list)
    return tf_df



def get_idf_df(data, columns=['context_lemma_pos', 'question_lemma_pos']):
    """computes a dataframe with inverse document frequency for all columns included"""
    corpus=[]
    for col in columns:
        doc=''
        for i in data[col]:
            for j in i:
                doc+=j[0]
                doc+=' '
                
        corpus.append(doc)
    cv = CountVectorizer()
    tfidf_transformer= TfidfTransformer(smooth_idf=True, use_idf=True)
    
    word_count_vector = cv.fit_transform(corpus)
    tfidf_transformer.fit(word_count_vector)
    
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
    df_idf.index = df_idf.index.map(str)

    return df_idf


def get_tf_idf_col(data, columns=['context_lemma_pos', 'question_lemma_pos']):
    """creates copy of dataframe and then iterates over named columns, updating 
    the word-tuple to include the token; POS-tag and the tf-idf tag. These values
    are stored in a new column."""
    ndf=data.copy()
    

    idf_df = get_idf_df(data, columns=columns)
    
    
    for col in columns:
        col_ind=0
        ndf['{col}_tfidf'.format(col=col)] = 0
        for i in ndf[col]:
        # here use row index as document index/ID and retrieve TFIDF score from the get_tf_df model
            new_col_term=[]
            tok_list = [x[0] for x in i]
            tf_df = get_tf_df(tok_list)
            for j in i:
                w = j[0]                
                w_pos = j[1]
                tf_score = tf_df['tf'][w]
                try:
                    idf_score = idf_df['idf_weights'][w]
                except KeyError:
                    idf_score = 1
                w_tfidf_score = tf_score*idf_score
                new_col_tuple=(w, w_pos, w_tfidf_score)
                new_col_term.append(new_col_tuple)
                
                # can and will get round to adding the actual tfidf score per word per do
                # rather than just the sum 
                
            ndf['{col}_tfidf'.format(col=col)][col_ind] = new_col_term
            col_ind+=1
            
    return ndf

#### WORD EMBEDDINGS RELATED FUNCTIONS

def get_total_vocab(data, column='text'):
    """Gets a list of all the tokens from the token-pos tuples in our columns 
    and adds them all to a single list"""
    vocab=[]
    
    for i in data[column]:
        j = i.split()
        for k in j:
            vocab.append(k)
    vocab.append('')
    set_vocab = set(vocab)
    vocab_dict = dict(zip(set_vocab, range(len(set_vocab))))
    
    return vocab_dict

def file_to_word_ids(vocabulary, data, input_col, output_col, 
                     max_input, max_output, pad = ''):
    txt = []
    word2id_dict = vocabulary
    
    data=data.reset_index(drop=True)
    
    discounted=0
    for i in data.index:
        try:    
            j = data.iloc[i][input_col]
            try:
                j = j.split(' ')
            except AttributeError:
                j = j
        except IndexError:
            print("Element not found at index ", i)
            j = data[input_col][i]
            try:
                j = j.split(' ')
            except AttributeError:
                j = j
                
        try:
            k = data.iloc[i][output_col]
            try:
                k = k.split(' ')
            except AttributeError:
                k = k
        except IndexError:
            print("Element not found at index ", i)
            k = data[output_col][i]
            try:
                k = k.split(' ')
            except AttributeError:
                k = k
            
        while len(j)<max_input:
            j.append(pad)
        while len(k)<max_output:
            k.append(pad)
        
        for w in j:
            try:
                txt.append(word2id_dict[w])
            except KeyError:
                continue
        for a in k:
            try:
                txt.append(word2id_dict[a])
            except KeyError:
                continue
    print(discounted)
    return txt


def get_glove_vectors(filepath, data, columns, vocab):
    glove_dict = {}
    total_vocab = vocab
    with open(file=filepath, mode='rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if word in total_vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                glove_dict[word] = vector
    return glove_dict

class W2vVectorizer(object):
    
    def __init__(self, w2v, mean_wv=True):
        # takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])
        self.mean_wv=mean_wv
    
    # Note from Mike: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # It can't be used in a sklearn Pipeline. 
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        if mean_wv:
            return np.array([
                   np.mean([self.w2v[w] for w in words if w in self.w2v]
                           or [np.zeros(self.dimensions)], axis=0) for words in X])