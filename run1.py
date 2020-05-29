import pandas as pd
import xgboost as xgb
import pickle
import sys
import scipy
from scipy import sparse
from scipy.sparse import hstack
import numpy as np
import io

from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import pickle

import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU
import numpy as np
import re


def convert(text):
    processed = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')
    # Replace urls with 'webaddress'
    processed = processed.str.replace(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', 'webaddress')
    processed = processed.str.replace(r'<[^>]+>', ' ')
    # Replace email addresses with 'emailaddr'
# Replace money symblos with 'moneysymb'
    processed = processed.str.replace(r'£|\$', 'moneysymb')
# Replace 10 digit phone numbers with 'phonenumber'
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]]{3}[\s-]?[\d]{4}$','phonenumber')
# Replace normal numbers with 'numbr'
    processed = processed.str.replace(r'\d+(\.\d+)?','numbr')
# Replace punctuation
    processed = processed.str.replace(r'[^\w\d\s]',' ')
# Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')
# Remove Leading and Trailing whitespaces
    processed = processed.str.replace(r'^\s+|\s+?$', '')
# Change words to lower case - Hello, HEllo, hello are all same word!
    processed = processed.str.lower()
    return (processed)

def convertS(text):
    # Replace email addresses with 'emailaddr'
    processed = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', 'webaddress', text)
    processed = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr', processed)
    processed = re.sub(r'<[^>]+>', ' ', processed)
    # Replace email addresses with 'emailaddr'

    # Replace urls with 'webaddress'
    # Replace money symblos with 'moneysymb'
    processed = re.sub(r'£|\$', 'moneysymb',processed)
    # Replace 10 digit phone numbers with 'phonenumber'
    processed = re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]]{3}[\s-]?[\d]{4}$', 'phonenumber',processed)
    # Replace normal numbers with 'numbr'
    processed = re.sub(r'\d+(\.\d+)?', 'numbr',processed)
    # Replace punctuation
    processed = re.sub(r'[^\w\d\s]', ' ',processed)
    # Replace whitespace between terms with a single space
    processed = re.sub(r'\s+', ' ',processed)
    # Remove Leading and Trailing whitespaces
    processed = re.sub(r'^\s+|\s+?$', '',processed)
    # Change words to lower case - Hello, HEllo, hello are all same word!
    processed = processed.lower()
    return (processed)

def preprocess_sample(df, tokenizer):
    df["Message"] = convert(df["Message"])
    df['Message'] = df['Message'].astype(str)
    df['isFriend'] = df['isFriend'].map({True: 1, False: 0})
    df['isBlocked'] = df['isBlocked'].map({True: 1, False: 0})
    df['len'] = df['Message'].apply(len)
    X = df['Message'].values
    df.drop(['Message', 'URL'], axis=1, inplace=True)
    tokenizer.texts_to_sequences(X)
    X_seq = tokenizer.texts_to_sequences(X)

    X_pad = pad_sequences(X_seq, maxlen=50, padding='post')
    X_pad = np.hstack((X_pad, df))
    X_pad = np.asarray(X_pad)

    return X_pad

def process_line(line, index):
    if index == 0:
        return str(line)[:-1]
    elif index == 1:
        return str(line)[:-1]
    elif index == 2:
        return line == 'True'
    elif index == 3:
        return int(line)
    elif index == 4:
        return int(line)
    elif index == 5:
        return float(line)
    elif index == 6:
        return int(line)
    elif index == 7:
        return int(line)
    elif index == 8:
        return line == 'True'

def main():

    tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

    model1 = keras.models.load_model('best_modelLSTM.h5')
    model2 = keras.models.load_model('best_modelGRU.h5')

    sys.stdin.reconfigure(encoding='utf-8')
    data = pd.DataFrame({'Message': pd.Series([], dtype='str'),
                         'URL': pd.Series([], dtype = 'str'),
                         'isFriend': pd.Series([], dtype = bool),
                         'Friends': pd.Series([], dtype=int),
                         'Followers': pd.Series([], dtype=int),
                         'RelationRatio': pd.Series([], dtype=float),
                         'PageAge': pd.Series([], dtype=int),
                         'Density': pd.Series([], dtype=int),
                         'isBlocked': pd.Series([], dtype=bool),})
    if len(sys.argv) != 10:
        print('wrond num of args')
        exit(-1)
    for index, line in enumerate(sys.argv[1::]):
        try:
            arg = process_line(line, index)
        except:
            print("args wrong")
            exit(-1)
        data.loc[0, data.columns[index]] = arg
    try:
        X = preprocess_sample(data, tokenizer)
        y_pred_lstm = model1.predict_proba(X)
        y_pred_gru = model2.predict_proba(X)
        print(y_pred_lstm[0])
        print(y_pred_gru[0])
    except:
        print('smth wrong, try one more time')
        exit(-1)

if __name__ == '__main__':
    main()


