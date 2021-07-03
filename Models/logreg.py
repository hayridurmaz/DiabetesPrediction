from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import printResults


def test_log_reg(X, y):
    print("TESTING: LOGISTIC REGRESSION")

    # Adjusting development threshold
    logreg_classifier = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logreg_classifier.fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg_classifier.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg_classifier.score(X_test, y_test)))

    # Predicting the Test set results
    y_pred = logreg_classifier.predict(X_test)

    printResults(y_test, y_pred)
