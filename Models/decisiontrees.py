from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import printResults


def test_decision_trees(X, y):
    print("TESTING: DECISION TREES")
    # Parameter evaluation
    treeclf = DecisionTreeClassifier(random_state=42)
    parameters = {'max_depth': [6, 7, 8, 9],
                  'min_samples_split': [2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]
                  }
    gridsearch = GridSearchCV(treeclf, parameters, cv=100, scoring='roc_auc')
    gridsearch.fit(X, y)
    print(gridsearch.best_params_)
    print(gridsearch.best_score_)

    # Adjusting development threshold
    tree = DecisionTreeClassifier(max_depth=6, max_features=4,
                                  min_samples_split=5,
                                  random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    tree.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

    # Predicting the Test set results
    y_pred = tree.predict(X_test)

    printResults(y_test, y_pred)
