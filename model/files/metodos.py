# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import cross_val_predict, KFold
# from sklearn.metrics import confusion_matrix, accuracy_score
 
import pandas as pd

f = open('data.csv')

# Se lee el archivo .csv con ayuda de Pandas
df = pd.read_csv(f)

# Por cada columna se van a etiquetar sus atributos para poder usarlos en los algoritmos.
for (name, data) in df.iteritems():
    df[name] = LabelEncoder().fit_transform(data)

# Se leen los datos de X y los datos de y
X = df.loc[:, df.columns != 'Class']
y = df.Class

# Tenemos 3 clasificadores diferentes. Se selecciona el que se desea
# clf = GaussianNB()
# clf = DecisionTreeClassifier("entropy")
clf = MLPClassifier(hidden_layer_sizes=(5,), solver="sgd", learning_rate_init=0.3, max_iter=200)

# Se hace la predicción KFold de 10
k_fold = KFold(len(y), n_splits=10)
y_predict = cross_val_predict(clf, X, y, cv=k_fold)

# Obtenemos los resultados de eficiencia y de la matriz de confusión
score = accuracy_score(y, y_predict)
matrix = confusion_matrix(y, y_predict)

# Imprimimos resultados
print(score)
print(matrix)
