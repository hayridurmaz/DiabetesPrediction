from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utils import printResults


def test_random_forest(X, y, X_train, y_train, X_test, y_test):
    print("TESTING: RANDOM FOREST")
    # Parameter evaluation
    rfclf = RandomForestClassifier(random_state=42)
    parameters = {'n_estimators': [50, 100],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [4, 5, 6, 7],
                  'criterion': ['gini', 'entropy']
                  }
    gridsearch = GridSearchCV(rfclf, parameters, cv=50, scoring='roc_auc', n_jobs=-1)
    gridsearch.fit(X, y)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)

    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=6,
                                max_features='auto', random_state=0)
    rf.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

    y_pred = rf.predict(X_test)

    printResults(y_test, y_pred)
