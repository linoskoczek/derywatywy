import sklearn as sk
import pandas as pd 
import morfeusz2
from hyphen import Hyphenator
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('dane.csv', sep=';')
data = data[['child_lemma', 'parent_lemma']]
data = data.drop_duplicates()
h_pl = Hyphenator('pl_PL')
pd_list = [[h_pl.syllables(c[0]), h_pl.syllables(c[1]), c[0], c[1]] for c in data.values]
pd_list = [[p[0][-1], p[1][-1], p[2], p[3]] for p in pd_list if p[0] != '' and p[0] != []]
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

algs = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
]

for neigh in algs:
    X = dfn[['m', 'ms']]
    y = dfn[['fs']]
    neigh.fit(X, y)

    predicted = neigh.predict(test[['m', 'ms']])
    pred=pd.DataFrame(predicted)
    pred.columns = ['fs']
    pred['fs'] = pred['fs'].apply(np.int64)
    pred = pred.rename(index=dict(zip(pred.index,test.index)))
    pred['m'] = test['m']
    pred['ms'] = test['ms']
    pred['c_f'] = test['f']
    pred['c_fs'] = test['fs']

    de = decode_df(pred, dict_le)

    print("Score for", type(neigh).__name__, accuracy_score(pred['fs'], test['fs']))

# for i in range(1, 500, 20):
#     print(de[i:i+20])