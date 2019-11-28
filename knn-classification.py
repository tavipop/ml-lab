import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('teleCust1000t.csv')
df.head()
df['custcat'].value_counts()
#df.hist(column='income', bins=50)
#df.columns

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

y = df['custcat'].values
y[0:5]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn import metrics

ks = 10
accs = []

for k in range(1, ks):
    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    #print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    acc = metrics.accuracy_score(y_test, yhat)
    print("Test set Accuracy:", k, acc)
    accs.append(acc)
accs

plt.plot(range(1,ks),accs, 'g')
#plt.fill_between(range(1,ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
#plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
#plt.tight_layout()
plt.show()

