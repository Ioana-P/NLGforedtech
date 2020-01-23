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
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding, Masking
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import argparse
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

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

def get_total_vocab(data, columns=['text']):
    """Gets a list of all the tokens from the token-pos tuples in our columns 
    and adds them all to a single list"""
    vocab=[]
    for column in columns:
        for i in data[column]:
            j = i.split()
            for k in j:
                vocab.append(k)
    vocab.append('')
    set_vocab = set(vocab)
    vocab_dict = dict(zip(set_vocab, range(1,len(set_vocab)+1)))
    vocab_dict.update({'---':0})
    return vocab_dict

def file_to_word_ids(vocabulary, data, input_col, output_col, 
                     max_input, max_output=0):
    word2id_dict = vocabulary
    txt_lst=[]
    data=data.reset_index(drop=True)
    
    if output_col==None:
        max_length=max_input
        data[input_col].drop_duplicates(inplace=True)
        
        for i in data[input_col]:
            try:
                j = i.split(' ')
            except AttributeError:
                j = i
            for w in j:
                if (w!='' and w!=' '):
                    try:
                        txt_lst.append(word2id_dict[w])
                    except KeyError:
                        raise KeyError('cannot find {}'.format(w))
                else:
                    continue
            
        txt_lst_final = [x for x in txt_lst if x!=2]        
            
    else:
        max_length = (max_input+max_output)
        for i in data.index:
            txt = []

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

            for w in j:
                if (w!='' and w!=' ' and w!=2):
                    try:
                        txt.append(word2id_dict[w])
                    except KeyError:
                        continue
                else:
                    continue
            for a in k:
                if (a!='' and a!=' ' and a!=2):
                    try:
                        txt.append(word2id_dict[a])
                    except KeyError:
                        continue
                else:
                    continue
            txt = [x for x in txt if x!=2]        
            txt_lst.append(txt)

        txt_lst_final = pad_sequences(txt_lst, maxlen=max_length, padding='pre')
    return txt_lst_final


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

def split_word_pos(word, sep='_'):
    w = ''
    p = ''
    try:
        for l in word[:word.index(sep)]:
            w+=l
        for t in word[word.index(sep)+1:]:
            p+=t
    except:
        w=word
        p=''
    return w, p

def get_pos_dict(df_w_pos, columns=['context_lemma_pos',	'question_lemma_pos',	'answers_lemma_pos'	]):
    pos_dict={}
    weight_list=[]
    df = df_w_pos.copy()
    pos_dict.update({'':0.0})
    for col in columns:
        for row in df[col]:
            for tup in row:
                new_pos = tup[1]
                if new_pos not in pos_dict.values():
                    new_weight = np.random.random()
                    while new_weight in weight_list:
                        new_weight = np.random.random()
                    pos_dict.update({new_pos: new_weight})
                    
    return pos_dict

def vectorize_pos(pos, pos_dict):
    pos_val = pos_dict[pos]
    return np.asarray(pos_val)
    
    
        
def load_data(data, pos_dict,
              use_glove=True, 
              embedding_dim=50,
              use_pos=False,
              train_split = 0.8,
              valid_split = 0.1,
              shuffle_df = True,
              input_col_name='text', output_col_name='answers_lstm',
              brief_printout=True):
    """ Takes in dataframe of untokenized text, expecting one input and one
    output column and transforms it into 3 arrays for training, validation 
    and testing data, each of dimensions:
    (word, max_sentence_length, vocabulary_size)
    ----------------------------------------------------------
    data - Pandas dataframe
    use_glove - bool, whether to use the pretrained Glove word embeddings
    downloaded from https://nlp.stanford.edu/projects/glove/
    embedding_dim - int, what is the dimension of the pretrained weights
    train_split - float<1, fraction of data to use as train
    valid_split - float<1, fraction of data to use for validation; test
    split is 1 - other splits summed.
    shuffle_df - whether to reorder the dataframe index
    input_col_name - str, name of dataframe column to retrieve 
    input text from
    output_col_name - str, name of dataframe column to retrieve 
    output text from
    brief_printout - bool, whether to provide a brief printout of the 
    results or not. 
    pos_dict - dict, k:v pairs of POS tags (e.g. JJ, $, VBP) and a stored random float
    ----------------------------------------------------------
    RETURNS:
    train_data, valid_data, test_data, - np.arrays
    vocabulary, reversed_dictionary, - dict
    vocabulary_size, int
    embedding_matrix, np.array
    embedding_dim, tuple(int)
    """
    
    data.reset_index(inplace=True, drop=True)
    
    if shuffle_df:
        data = data.sample(frac=1).reset_index(drop=True)
    
    train_df = data.iloc[:round(train_split*len(data))]
    valid_df = data.iloc[round(train_split*len(data)):round((1-valid_split)*len(data))]
    test_df = data.iloc[round((train_split+valid_split)*len(data)):]
    
    input_max_len = data[input_col_name].apply(lambda x: len(x.split(' '))).max()
    output_max_len = data[output_col_name].apply(lambda x: len(x.split(' '))).max()

    # build the complete vocabulary, then convert text data to dict of integer-word pairs
    vocabulary = get_total_vocab(data, columns=input_col_name)
    train_data = file_to_word_ids(vocabulary, train_df, input_col=input_col_name, output_col = output_col_name, 
                                      max_input=input_max_len, max_output=output_max_len)
    valid_data = file_to_word_ids(vocabulary, valid_df, input_col=input_col_name, output_col = output_col_name, 
                                      max_input=input_max_len, max_output=output_max_len)
    test_data = file_to_word_ids(vocabulary, test_df, input_col=input_col_name, output_col = output_col_name, 
                                      max_input=input_max_len, max_output=output_max_len)
    vocabulary_size = len(vocabulary)
    reversed_dictionary = dict(zip(vocabulary.values(), vocabulary.keys()))
    
    if use_glove:
        if use_pos:
            pos_dim = 1
           
        else:
            pos_dim = 0
        dim=embedding_dim
        if dim not in [50, 100, 200, 300]:
            raise Exception('Specified embedding dimension ({}) not available. Current GloVe embeddings \n are [50, 100, 200, 300].'.format(dim))
        embeddings_index = {}
        f = open(os.path.join('glove.6B/glove.6B.{}d.txt'.format(dim)))
        coefs_sum=np.zeros((dim,))
        coefs_count = 0
        # here we extract and build a dict containing all the words from the GloVe file and stor them under embeddings_index
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            coefs_sum += coefs
            coefs_count+=1
        f.close()
        coefs_mean = coefs_sum/coefs_count
        
        # the emb_matrix will be where we actually store our vectors; if using pos
        # then we split the matrix: for each of the unique word&pos combinations
        # we take let the first "dim" nr elements be taken up the embedding vector
        # 
        embedding_matrix = np.zeros((len(vocabulary), dim+pos_dim))
        for word, i in vocabulary.items():
            if use_pos:
                word, pos = split_word_pos(word)
                unique_pos_vector = vectorize_pos(pos, pos_dict)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i,:dim] = embedding_vector
                if use_pos:    
                    embedding_matrix[i,dim:]=unique_pos_vector
            else:
                embedding_matrix[i,:dim] = coefs_mean*np.random.random()
                if use_pos:
                    embedding_matrix[i,dim:]=unique_pos_vector 
    else:
        embedding_matrix = None
        embedding_dim = None
    
    one = reversed_dictionary[np.random.randint(2, vocabulary_size)]
    two = reversed_dictionary[np.random.randint(2, vocabulary_size)]
    three = reversed_dictionary[np.random.randint(2, vocabulary_size)]
    
    
    if brief_printout:
        print("First five train sentences in vectorized format : ", train_data[:5])
        print("Vocabulary examples : ")
        print("1. ", one, vocabulary[one])
        print("2. ", two, vocabulary[two])
        print("3. ", three, vocabulary[three])
        print(" Size of vocabulary : ", vocabulary_size)
        train_data_lst=train_data.tolist()
        print(" ".join([reversed_dictionary[x] for x in train_data_lst[10] if x!=0]))
        print("")
    
    
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary, vocabulary_size, embedding_matrix, embedding_dim+pos_dim


def load_sequential_data(data, pos_dict,
              use_glove=True, 
              embedding_dim=50,
              use_pos=False,
              train_split = 0.9,
              valid_split = 0.1,
              shuffle_df = True,
              input_col_name='context_lemma_pos',
              output_test_col_name='question_answer',
              vocab_col_names=['text'],
              brief_printout=True):
    """ Takes in dataframe of untokenized text, expecting one input and one
    output column and transforms it into 3 arrays for training, validation 
    and testing data, each of dimensions:
    (word, max_sentence_length, vocabulary_size)
    ----------------------------------------------------------
    data - Pandas dataframe
    use_glove - bool, whether to use the pretrained Glove word embeddings
    downloaded from https://nlp.stanford.edu/projects/glove/
    embedding_dim - int, what is the dimension of the pretrained weights
    train_split - float<1, fraction of data to use as train
    valid_split - float<1, fraction of data to use for validation; test
    split is 1 - other splits summed.
    shuffle_df - whether to reorder the dataframe index
    input_col_name - str, name of dataframe column to retrieve 
    input text from
    output_col_name - str, name of dataframe column to retrieve 
    output text from
    brief_printout - bool, whether to provide a brief printout of the 
    results or not. 
    pos_dict - dict, k:v pairs of POS tags (e.g. JJ, $, VBP) and a stored random float
    ----------------------------------------------------------
    RETURNS:
    train_data, valid_data, test_data, - np.arrays
    vocabulary, reversed_dictionary, - dict
    vocabulary_size, int
    embedding_matrix, np.array
    embedding_dim, tuple(int)
    """
    
    data.reset_index(inplace=True, drop=True)
    
    if shuffle_df:
        data = data.sample(frac=1).reset_index(drop=True)
    
    train_df = data.iloc[:round(train_split*len(data))]
    valid_df = data.iloc[round(train_split*len(data)):]
#     test_df = data.iloc[round((train_split+valid_split)*len(data)):]
    
    input_max_len = data[input_col_name].apply(lambda x: len(x.split(' '))).max()
#     output_max_len = data[output_col_name].apply(lambda x: len(x.split(' '))).max()

    # build the complete vocabulary, then convert text data to dict of integer-word pairs
    vocabulary = get_total_vocab(data, columns=vocab_col_names)
    train_data = file_to_word_ids(vocabulary, train_df, input_col=input_col_name, output_col = None, 
                                      max_input=input_max_len)
    valid_data = file_to_word_ids(vocabulary, valid_df, input_col=input_col_name, output_col = None, 
                                      max_input=input_max_len)
#     test_data = file_to_word_ids(vocabulary, test_df, input_col=input_col_name, output_col = None, 
#                                       max_input=input_max_len)
    vocabulary_size = len(vocabulary)
    reversed_dictionary = dict(zip(vocabulary.values(), vocabulary.keys()))
    
    if use_glove:
        if use_pos:
            pos_dim = 1
           
        else:
            pos_dim = 0
        dim=embedding_dim
        if dim not in [50, 100, 200, 300]:
            raise Exception('Specified embedding dimension ({}) not available. \n Current GloVe embeddings are [50, 100, 200, 300].'.format(dim))
        embeddings_index = {}
        f = open(os.path.join('glove.6B/glove.6B.{}d.txt'.format(dim)))
        coefs_sum=np.zeros((dim,))
        coefs_count = 0
        # here we extract and build a dict containing all the words from the GloVe file and stor them under embeddings_index
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            coefs_sum += coefs
            coefs_count+=1
        f.close()
        coefs_mean = coefs_sum/coefs_count
        
        # the emb_matrix will be where we actually store our vectors; if using pos
        # then we split the matrix: for each of the unique word&pos combinations
        # we take let the first "dim" nr elements be taken up the embedding vector
        # 
        embedding_matrix = np.zeros((len(vocabulary), dim+pos_dim))
        for word, i in vocabulary.items():
            if use_pos:
                word, pos = split_word_pos(word)
                unique_pos_vector = vectorize_pos(pos, pos_dict)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i,:dim] = embedding_vector
                if use_pos:    
                    embedding_matrix[i,dim:]=unique_pos_vector
            else:
                embedding_matrix[i,:dim] = coefs_mean*np.random.random()
                if use_pos:
                    embedding_matrix[i,dim:]=unique_pos_vector 
    else:
        embedding_matrix = None
        embedding_dim = None
    
    one = reversed_dictionary[np.random.randint(2, vocabulary_size)]
    two = reversed_dictionary[np.random.randint(2, vocabulary_size)]
    three = reversed_dictionary[np.random.randint(2, vocabulary_size)]
    
    
    if brief_printout:
        print("First five words  in vectorized format : ", train_data[:5])
        print("Vocabulary examples : ")
        print("1. ", one, vocabulary[one])
        print("2. ", two, vocabulary[two])
        print("3. ", three, vocabulary[three])
        print(" Size of vocabulary : ", vocabulary_size)
#         train_data_lst=train_data.tolist()
        print(" ".join([reversed_dictionary[x] for x in train_data[0:30] if x!=0]))
        print("")
    
    
    return train_data, valid_data, vocabulary, reversed_dictionary, vocabulary_size, embedding_matrix, embedding_dim+pos_dim



class KerasBatchGenerator(object):
    """Object that takes in numpy array data, number of 'steps' to take 
    along sequence of data for each iteration; batch-size, 
    ----------------------------------------------------------
    data - np.arrays
    num_steps - int, how long one input sequence should be
    vocabulary - dict, k:v word : unique integer pairs
    skip_step - int, optional - how many sequence elements to skip
    after each iteration
    answer_length - how long the output sequence is; batch generator 
    will take from the end of input sequence
    ----------------------------------------------------------
    YIELDS:
    x, y - tuple of equal sized np.arrays, to be fed into a Keras
    model
    """
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step = 0, answer_length=1):
        """Object that takes in numpy array data, number of 'steps' to take 
        along sequence of data for each iteration; batch-size, 
        ----------------------------------------------------------
        data - np.arrays
        num_steps - int, how long one input sequence should be
        vocabulary - dict, k:v word : unique integer pairs
        skip_step - int, optional - how many sequence elements to skip
        after each iteration
        answer_length - how long the output sequence is; batch generator 
        will take from the end of input sequence
        ----------------------------------------------------------
        YIELDS:
        x, y - tuple of equal sized np.arrays, to be fed into a Keras
        model
        """
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.skip_step = skip_step
        self.vocabulary_size = len(vocabulary)
        self.answer_length = answer_length
        
        
    #     
    #     skip_steps = how far to move the window after the prediction is made

    def generate(self, one_sequence=False):
        """Object that takes in numpy array data, number of 'steps' to take 
        along sequence of data for each iteration; batch-size, 
        ----------------------------------------------------------
        data - np.arrays
        num_steps - int, how long one input sequence should be
        vocabulary - dict, k:v word : unique integer pairs
        skip_step - int, optional - how many sequence elements to skip
        after each iteration
        answer_length - how long the output sequence is; batch generator 
        will take from the end of input sequence
        ----------------------------------------------------------
        YIELDS:
        x, y - tuple of equal sized np.arrays, to be fed into a Keras
        model
        """
        x = np.zeros((self.batch_size, self.num_steps)) # input nn nodes, same dimensions as batchsize x nr of words in 1 window
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary_size)) # output nodes
        if one_sequence:
            while True:
                for i in range(self.batch_size):
                    if self.current_idx + self.num_steps >= len(self.data):
                        # reset the index back to the start of the data set
                        self.current_idx = 0
                    x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                    temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                    # convert all of temp_y into a one hot representation
                    y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary_size)
                    self.current_idx += self.skip_step
            
                yield x,y
        else:
            while True:
                for i in range(self.batch_size):
                    if self.current_idx + self.num_steps >= len(self.data): # once the ticker goes over the length of data, reset it
                        self.current_idx = 0
                    x[i,:] = self.data[self.current_idx][:-self.answer_length]
                    temp_y = self.data[self.current_idx][self.answer_length:]
                    # converts all the output y into a 1-hot representation
                    y[i,:, :] = to_categorical(temp_y, num_classes=self.vocabulary_size)
                    self.current_idx+=1


                yield x, y
            

            
def concat_tuple(lst):
    result=''
    word_pos=''
    for x in lst:
#         for y in x:
#             word_pos+=str(y)+'_'
        word_pos = str(x[0])+'_'+str(x[1])
        word_pos = word_pos.replace('\n', '')
        word_pos = (word_pos.encode('ascii', 'ignore')).decode("utf-8")
        
        result+= ' ' + word_pos
    result = result[1:]
    
    return result
            
def concat_word_pos(df, columns= ['context', 'answers']):
    ndf = df.copy()
    
    for col in columns:
        ndf[col] = ndf[col].apply(lambda tup: concat_tuple(tup))
    
    return ndf