import warnings

from Models.decisiontrees import test_decision_trees
from Models.knn import tes_knn
from Models.logreg import test_log_reg
from Models.neural_nets import test_nn
from Models.randomforest import test_random_forest
from Models.svm import test_svm
from utils import getDataset

warnings.filterwarnings('ignore')


def main():
    X, y, X_train, X_test, y_train, y_test, dataset = getDataset()
    test_nn(dataset)
    test_decision_trees(X, y)
    tes_knn(X, y, X_train, y_train, X_test, y_test)
    test_log_reg(X, y)
    test_random_forest(X, y, X_train, y_train, X_test, y_test)
    test_svm(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
