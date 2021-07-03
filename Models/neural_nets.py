import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def test_nn(dataset):
    print("TESTING: NN")
    np.random.seed(10)
    Z = dataset[:, 0:8]
    Q = dataset[:, 8]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])

    model.fit(Z, Q, epochs=150, batch_size=10)

    predictions = model.predict(Z)

    rounded = [round(x[0]) for x in predictions]
    print(rounded)

    scores = model.evaluate(Z, Q)
    i = 0
    for s in scores:
        print("\n%s: %.2f%%" % (model.metrics_names[i], s))
        i = i + 1
