import sklearn as sk
import sys
import pandas as pd
from hyphen import Hyphenator
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import operator
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('dane.csv', sep=';')
data = data[['child_lemma', 'parent_lemma']]
data = data.drop_duplicates()
h_pl = Hyphenator('pl_PL')
pd_list = [[c[0][-5:], c[1][-5:], c[0], c[1]] for c in data.values]
df = pd.DataFrame().from_records(pd_list)
df.columns = ['ms', 'fs', 'm', 'f']


def encode_df(dataframe, le):
    dataframe = dataframe.copy()
    for column in dataframe.columns:
        dataframe[column] = le[column].fit_transform(dataframe[column])
    return dataframe


def decode_df(dataframe, le):
    dataframe = dataframe.copy()
    for column in dataframe.columns:
        dataframe[column] = le[column].inverse_transform(dataframe[column])
    return dataframe


fs_le = preprocessing.LabelEncoder()
f_le = preprocessing.LabelEncoder()

dict_le = {
    'm': preprocessing.LabelEncoder(),
    'ms': preprocessing.LabelEncoder(),
    'fs': fs_le,
    'c_fs': fs_le,
    'f': f_le,
    'c_f': f_le
}
dfn = encode_df(df, dict_le)

train, test = train_test_split(dfn, test_size=0.2)

def find_best_algorithm():
    best_results = None
    best_score = 0

    algs = [
        LinearRegression(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        KNeighborsClassifier(),
        DecisionTreeClassifier()
    ]

    scores = {}

    for alg in algs:
        X = dfn[['m', 'ms']]
        y = dfn[['fs']]
        alg.fit(X, np.ravel(y))

        predicted = alg.predict(test[['m', 'ms']])
        pred = pd.DataFrame(predicted)
        pred.columns = ['fs']
        pred['fs'] = pred['fs'].apply(np.int64)
        pred = pred.rename(index=dict(zip(pred.index, test.index)))
        pred['m'] = test['m']
        pred['ms'] = test['ms']
        pred['c_f'] = test['f']
        pred['c_fs'] = test['fs']

        decoded_results = decode_df(pred, dict_le)
        score = accuracy_score(pred['fs'], test['fs'])
        scores[score] = decoded_results
        print('{:>15} {}'.format(type(alg).__name__, score))
        if score > best_score:
            best_score = score
            best_results = decoded_results

    return best_results

def calc_str_offset(str1, str2):
    f = str1.find(str2[0])
    if f != -1:
        return len(str1) - f
    else:
        return len(str1)


decoded_results = find_best_algorithm()
mfs = list(decoded_results[['m', 'ms', 'fs']].values)

final_results = []
for r in mfs:
    m = r[0][:-calc_str_offset(r[1], r[2])]
    final_results.append(m + r[2])
    #print(r[0], '->', m + r[2])

decoded_results['final_results'] = final_results
decoded_results['guess_ok'] = np.where(decoded_results['final_results'] == decoded_results['c_f'], True, False)
count_true = decoded_results.guess_ok[decoded_results.guess_ok == True].count()
count_all = len(decoded_results.index)
print(decoded_results[['m', 'final_results', 'guess_ok']].head(-5))
print("\nFinal accuracy:", count_true, "/", count_all, '->', count_true/count_all * 100, '%')