import sklearn as sk
import sys
import pandas as pd
import operator
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(0.1)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


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


def teach_model_and_test(alg, dfn, train, test, dict_le, letter_number):
    X = train[['m', 'ms']]
    y = train[['fs']]
    alg.fit(X, np.ravel(y))

    # fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    # X, y = load_digits(return_X_y=True)
    # plot_learning_curve(alg, "Learning curve", X, y, ylim=(0.7, 1.01), n_jobs=4)
    # plt.savefig('plots/' + type(alg).__name__ + '-' + str(letter_number) + '.png')
    # plt.close('all')

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
    score = accuracy_score(test['fs'], pred['fs'])

    # if letter_number == 3 and type(alg).__name__ == 'DecisionTreeClassifier':
    #     skplt.metrics.plot_confusion_matrix(test['fs'], pred['fs'], normalize=True, hide_counts=True, hide_zeros=True)
    #     plt.show()

    return score, decoded_results


def find_best_algorithm_and_return_result(dfn, train, test, dict_le, letter_number):
    best_results = None
    best_alg = None
    best_score = 0

    algs = [
        LinearRegression(),
        KNeighborsRegressor(),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        # DecisionTreeClassifier(max_depth=15),
    ]

    scores = {}

    objects = tuple([str(type(alg).__name__) for alg in algs])
    y_pos = np.arange(len(objects))
    performance = []

    for alg in algs:
        score, decoded_results = teach_model_and_test(
            alg, dfn, train, test, dict_le, letter_number)
        scores[score] = decoded_results
        print('{:>15} {}'.format(type(alg).__name__, score))
        if score > best_score:
            best_score = score
            best_results = decoded_results
            best_alg = alg
        performance.append(score)

    # if letter_number == 3:
    #     plt.bar(y_pos, performance, align='center', alpha=0.5)
    #     plt.xticks(y_pos, objects)
    #     plt.ylabel('Accuracy')
    #     plt.xlabel('Algorithm')
    #     plt.title('Results with different algorithms')
    #     plt.show()
    return best_results, best_alg


def calc_str_offset(str1, str2):
    f = str1.find(str2[0])
    if f != -1:
        return len(str1) - f
    else:
        return len(str1)


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
    count_true = decoded_results.guess_ok[decoded_results.guess_ok == True].count(
    )
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
        pd_list = [[c[0][-letter_number:], c[1][-letter_number:], c[0], c[1]]
                   for c in data.values]
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

        decoded_results, alg = find_best_algorithm_and_return_result(
            dfn, train, test, dict_le, letter_number)
        decoded_results = merge_endings(decoded_results)
        score, count_true, count_all = calculate_score(decoded_results)
        if score > best_score:
            best_score = score
            best_count_true = count_true
            best_letter_number = letter_number
            best_results = decoded_results
        print("Result:", score, '%')

    return best_count_true, count_all, best_score, best_letter_number, best_results


count_true, count_all, score, best_letter_number, best_results = main()
print("\nFinal accuracy:", count_true, "/", count_all, '->',
      score, '%', 'for', best_letter_number, 'letters')
print(best_results[['m', 'final_results', 'guess_ok']].head(15))
