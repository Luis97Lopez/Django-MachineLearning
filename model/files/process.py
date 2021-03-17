import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def make():
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'data.csv')
    f = open(file_path)
    df = pd.read_csv(f)

    # Por cada columna se van a etiquetar sus atributos para poder usarlos en los algoritmos.
    for (name, data) in df.iteritems():
        df[name] = LabelEncoder().fit_transform(data)
    
    # Se leen los datos de X y los datos de y
    X = df.loc[:, df.columns != 'Class']
    y = df.Class

    # Clasificador del modelo.
    # clf = GaussianNB()
    clf = MLPClassifier(hidden_layer_sizes=(5,), solver="sgd", learning_rate_init=0.3, max_iter=200)
    
    # Se hace la predicción KFold de 10
    y_predict = cross_val_predict(clf, X, y, cv=10)

    # Obtenemos los resultados de eficiencia y de la matriz de confusión
    score = accuracy_score(y, y_predict)
    matrix = confusion_matrix(y, y_predict)

    # Imprimimos resultados
    print(type(X))
    print(type(y))
    print(type(score))
    print(type(matrix))

    return X