
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



mystem = Mystem()
russian_stopwords = stopwords.words("russian")
nltk.download("stopwords")

# NGRAM_RANGE = (2, 4)
# # слова н-граммы
# TOKEN_MODE = 'word'
# # максимальное кол-во фичей
# TOP_K = 1000
# # минимальная частота
# MIN_DOCUMENT_FREQUENCY = 2
# #

def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return text

def ngram_sample(texts, vectorizer, selector):
    x_val = vectorizer.transform(texts)
    x_val = selector.transform(x_val).astype('float32')
    return x_val

def preprocess_sample(data, vectorizer, selector):
    try:
        data_new = data.copy()
        data_new['isFriend'] = data_new['isFriend'].map({True: 1, False: 0})
        data_new['isBlocked'] = data_new['isBlocked'].map({True: 1, False: 0})
        data_new.drop(['Density'], axis = 1, inplace = True)
        data_new['len'] = data_new['Message'].apply(len)
        data_new['Message'] = data_new['Message'].apply(preprocess_text)
        X_train_c = ngram_sample(data_new['Message'], vectorizer, selector)
        nn = data_new.drop(['Message', 'URL'], axis=1)
        print(nn)
        new_nn = hstack([nn, X_train_c]).toarray()
        print(new_nn)
        return new_nn
    except:
        print('problems with preprocessing')
        return new_nn

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
    sys.stdin.reconfigure(encoding='utf-8')
    vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    selector = pickle.load(open('selector.pickle', 'rb'))
    model = pickle.load(open('xgb.model', "rb"))
    print(model)
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
        X = preprocess_sample(data, vectorizer, selector)
        y_pred = model.predict_proba(X)
        y_p = model.predict(X)
        print(y_pred[0])
        print(y_p)
    except:
        print('smth wrong, try one more time')
        exit(-1)

if __name__ == '__main__':
    main()


