from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils import printResults


def test_svm(X_train, y_train, X_test, y_test):
    print("TESTING: SVM")
    # svm with grid search
    global y_pred
    svm = SVC(random_state=42)
    parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75),
                  'gamma': (1, 2, 3, 'auto'), 'decision_function_shape': ('ovo', 'ovr'),
                  'shrinking': (True, False)}

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        svm = GridSearchCV(SVC(), parameters, cv=5,
                           scoring='%s_macro' % score)
        svm.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(svm.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = svm.cv_results_['mean_test_score']
        stds = svm.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, svm.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, svm.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    svm_model = SVC(kernel='rbf', C=100, gamma=0.0001, random_state=42)
    svm_model.fit(X_train, y_train)
    spred = svm_model.predict(X_test)
    print('Accuracy with SVM {0}'.format(accuracy_score(spred, y_test) * 100))

    printResults(y_test, y_pred)
