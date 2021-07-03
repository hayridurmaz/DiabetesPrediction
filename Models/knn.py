from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from utils import printResults


def tes_knn(X, y, X_train, y_train, X_test, y_test):
    print("TESTING: KNN")
    # Parameter evaluation
    knnclf = KNeighborsClassifier()
    parameters = {'n_neighbors': range(1, 20)}
    gridsearch = GridSearchCV(knnclf, parameters, cv=100, scoring='roc_auc')
    gridsearch.fit(X, y)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)

    # Fitting K-NN to the Training set
    knnClassifier = KNeighborsClassifier(n_neighbors=18)
    knnClassifier.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knnClassifier.score(X_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knnClassifier.score(X_test, y_test)))

    # Predicting the Test set results
    y_pred = knnClassifier.predict(X_test)

    printResults(y_test, y_pred)
