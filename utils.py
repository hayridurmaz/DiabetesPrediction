import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


def printResults(y_test, y_pred):
    try:
        cm = confusion_matrix(y_test, y_pred)
        print('TP - True Negative {}'.format(cm[0, 0]))
        print('FP - False Positive {}'.format(cm[0, 1]))
        print('FN - False Negative {}'.format(cm[1, 0]))
        print('TP - True Positive {}'.format(cm[1, 1]))
        print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0, 0], cm[1, 1]]), np.sum(cm))))
        print('Misclassification Rate: {}\n'.format(np.divide(np.sum([cm[0, 1], cm[1, 0]]), np.sum(cm))))
        round(roc_auc_score(y_test, y_pred), 5)
    except Exception as e:
        print("Something went wrong.")
    try:
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print("Something went wrong.")


def getDataset():
    # Importing the dataset
    dataset = pd.read_csv('./Dataset/diabetes.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X, y, X_train, X_test, y_train, y_test, np.loadtxt('./Dataset/diabetes_without_header.csv', delimiter=",")
