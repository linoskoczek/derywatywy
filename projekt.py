import sklearn as sk
import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import operator
import numpy as np
from sklearn.model_selection import train_test_split


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

def teach_model_and_test(alg, dfn, train, test, dict_le):
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
    score = accuracy_score(test['fs'], pred['fs'], normalize=True)
    return score, decoded_results

def find_best_algorithm_and_return_result(dfn, train, test, dict_le):
    best_results = None
    best_alg = None
    best_score = 0

    algs = [
        LinearRegression(),
        KNeighborsRegressor(),
        KNeighborsClassifier(),
        # DecisionTreeRegressor(),
        # DecisionTreeClassifier()
    ]

    scores = {}
    for alg in algs:
        score, decoded_results = teach_model_and_test(alg, dfn, train, test, dict_le)
        scores[score] = decoded_results
        print('{:>15} {}'.format(type(alg).__name__, score))
        if score > best_score:
            best_score = score
            best_results = decoded_results
            best_alg = alg

    return best_results, best_alg


def calc_str_offset(str1, str2):
    f = str1.find(str2[0])
    if f != -1:
        return len(str1) - f
    else:
        return len(str1)


def manual_test(alg, dfn, dict_le):
    train = ['test']
    score, decoded_results = teach_model_and_test(alg, dfn, train, test, dict_le)
    decoded_results = merge_endings(decoded_results)
    return score, decoded_results


def merge_endings(decoded_results):
    mfs = list(decoded_results[['m', 'ms', 'fs']].values)

    final_results = []
    for r in mfs:
        m = r[0][:-calc_str_offset(r[1], r[2])]
        final_results.append(m + r[2])

    decoded_results['final_results'] = final_results
    decoded_results['guess_ok'] = np.where(
        decoded_results['final_results'] == decoded_results['c_f'], True, False)
    return decoded_results


def calculate_score(decoded_results):
    count_true = decoded_results.guess_ok[decoded_results.guess_ok == True].count()
    count_all = len(decoded_results.index)
    score = count_true/count_all * 100
    return score, count_true, count_all


def main():
    best_letter_number = 0
    best_score = 0
    best_count_true = 0
    best_results = None
    
    data = pd.read_csv('dane.csv', sep=';')
    data = data[['child_lemma', 'parent_lemma']]
    data = data.drop_duplicates()

    count_all = len(data)

    for letter_number in range(1, 10):
        pd_list = [[c[0][-letter_number:], c[1][-letter_number:], c[0], c[1]] for c in data.values]
        df = pd.DataFrame().from_records(pd_list)
        df.columns = ['ms', 'fs', 'm', 'f']

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
        print('\n[TEST FOR LEARNING WITH LAST', letter_number, 'LETTERS]')

        train, test = train_test_split(dfn, test_size=0.2)

        decoded_results, alg = find_best_algorithm_and_return_result(dfn, train, test, dict_le)
        decoded_results = merge_endings(decoded_results)
        score, count_true, count_all = calculate_score(decoded_results)
        if score > best_score:
            best_score = score
            best_count_true = count_true
            best_letter_number = letter_number
            best_results = decoded_results
        print("Result:", score, '%')
    
    # score, decoded_results = manual_test(alg, dfn, dict_le)
    # print('manual:', score)
    
    return best_count_true, count_all, best_score, best_letter_number, best_results

count_true, count_all, score, best_letter_number, best_results = main()
print("\nFinal accuracy:", count_true, "/", count_all, '->', score, '%', 'for', best_letter_number, 'letters')
print(best_results[['m', 'final_results', 'guess_ok']].head(15))