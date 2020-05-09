# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:32:48 2019

@author: LENOVO
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

df =pd.read_csv('Mall_Customers.csv',)
X = df.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as sch
d = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters =5, affinity='euclidean', linkage= 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s =100, c ='red')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s =100, c ='green')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s =100, c ='cyan')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s =100, c ='yellow')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s =100, c ='blue')
plt.ylabel('salary')
plt.xlabel('Spending Score')
plt.legend()
plt.show()







#Classification


from sklearn.preprocessing import StandardScaler

X_C = StandardScaler().fit_transform(X)


from sklearn.model_selection import train_test_split

df1 =df

X_train,X_test,y_train,y_test = train_test_split(X_C,y_hc,train_size=0.8,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn =KNeighborsClassifier(n_neighbors=5,p=2,metric= 'minkowski')

knn.fit(X_train,y_train)

from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def print_score(knn,X_train,y_train,X_test,y_test,train=True):
    if train:
        print('Training Result:\n')
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,knn.predict(X_train))))
        print('Classification:\n {}\n'.format(classification_report(y_train,knn.predict(X_train))))
        print('Confusion Matrix: \n {}\n'.format(confusion_matrix(y_train,knn.predict(X_train))))
        
        res = cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}'.format(np.mean(res)))
        print('Accuracy SD: \t{0:.4f}'.format(np.std(res)))
        
    elif train==False:
        print('Testing Result:\n')
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,knn.predict(X_test))))
        print('Classification:\n {}\n'.format(classification_report(y_test,knn.predict(X_test))))
        print('Confusion Matrix: \n {}\n'.format(confusion_matrix(y_test,knn.predict(X_test))))
   
print_score(knn,X_train,y_train,X_test,y_test,train=True)


print_score(knn,X_train,y_train,X_test,y_test,train=False)              

y_pred=knn.predict(X_test)



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','cyan','yellow','blue'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Spending score')
plt.ylabel(' Salary')
plt.legend()
plt.show()
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','cyan','yellow','blue'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Spending score')
plt.ylabel(' Salary')
plt.legend()
plt.show()
