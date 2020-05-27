
import pandas as pd
from pymystem3 import Mystem
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import pickle
import numpy as np



mystem = Mystem() 
russian_stopwords = stopwords.words("russian")
nltk.download("stopwords")

NGRAM_RANGE = (1, 5)
# слова н-граммы
TOKEN_MODE = 'char'
# максимальное кол-во фичей
TOP_K = 1000
# минимальная частота
MIN_DOCUMENT_FREQUENCY = 2


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return text

def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'lowercase': True,
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    print("start")
    vectorizer = TfidfVectorizer(**kwargs)
    print("vectorizer_built")
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val, vectorizer, selector

def preprocess_data(path_to_data):
    data = pd.read_excel(path_to_data)
    data.drop(['Density'], axis = 1, inplace = True)
    data.dropna(inplace = True)
    data['len'] = data['Message'].apply(len)
    print(data.columns)
    data['isSpam'] = data['isSpam'].map({True: 1, False: 0})
    data['isFriend'] = data['isFriend'].map({True: 1, False: 0})
    data['isBlocked'] = data['isBlocked'].map({True: 1, False: 0})
    data['Message'] = data['Message'].apply(preprocess_text)


    X = data.drop(['isSpam'], axis=1)
    y = data['isSpam']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
    stratify = y, test_size=0.1)
    X_train_c, X_test_c, vectorizer, selector = ngram_vectorize(X_train['Message'], y_train, X_test['Message'])
    nn = X_train.drop(['URL','Message'], axis=1)
    nn_c = X_test.drop(['URL','Message'], axis=1)
    # new_nn = hstack([nn, X_train_c]).toarray()
    # new_nnc = hstack([nn_c, X_test_c]).toarray()
    new_nn = X_train_c
    new_nnc = X_test_c
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    pickle.dump(selector, open("selector.pickle", "wb"))
    return new_nn, new_nnc, y_train, y_test, vectorizer, selector


def main():
    X_train, X_test, y_train, y_test, vectorizer, selector = preprocess_data('last_data.xlsx')
    model = xgb.XGBClassifier(random_state=42, learning_rate=0.001)
    model.fit(X_train, y_train)
    print(model.feature_importances_)
    y_pred = model.predict(X_test)
    print('new_test')
    data_n = pd.DataFrame({'Message': pd.Series(['Дорогие друзья , не надо стесняться, предоставляю бесплатный доступ только сегодня на сайт выигрыш'], dtype='str'),
                         'URL': pd.Series(['vk'], dtype = 'str'),
                         'isFriend': pd.Series([False], dtype = bool),
                         'Friends': pd.Series([0], dtype=int),
                         'Followers': pd.Series([0], dtype=int),
                         'RelationRatio': pd.Series([10.62786], dtype=float),
                         'PageAge': pd.Series([0], dtype=int),
                         # 'Density': pd.Series([0], dtype=int),
                         'isBlocked': pd.Series([True], dtype=bool),})
    data_n.drop(['Density'], axis=1, inplace=True)
    data_n.dropna(inplace=True)
    data_n['len'] = data_n['Message'].apply(len)
    print(data_n.columns)
    # data_n['isSpam'] = data_n['isSpam'].map({True: 1, False: 0})
    data_n['isFriend'] = data_n['isFriend'].map({True: 1, False: 0})
    data_n['isBlocked'] = data_n['isBlocked'].map({True: 1, False: 0})
    data_n['Message'] = data_n['Message'].apply(preprocess_text)
    Xx = data_n
    x_message = vectorizer.transform(Xx['Message'])
    x_message = selector.transform(x_message).astype('float32')
    nn = Xx.drop(['URL','Message'], axis=1)
    new_nn = hstack([nn, x_message]).toarray()
    yyy = model.predict_proba(x_message)
    print('!!!', yyy)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    print('acc: ', accuracy_score(y_test, y_pred))
    print('auc_roc: ', roc_auc_score(y_test, y_pred))
    pickle.dump(model, open('xgb.model', "wb"))

if __name__ == '__main__':
    main()

