# Importing essential libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from skewedmetrics import precision
from sklearn import linear_model

# Data visualisation
data = pd.read_csv("Dataset/creditcard.csv")
data = data.dropna( axis =0)
X = data.loc[:,'Time':'Amount']
y = data['Class']

clf = linear_model.SGDClassifier(max_iter=1000, tol= 1e-3)
clf.fit( X,y)
ytrue = y == 1
#print(ytrue)
ypred = clf.predict(X[ytrue])

count = (y[ytrue] == ypred).value_counts()
print(count)
print( "ans is %f" %( precision( y[ytrue], ypred) ))

'''ndata = data[data['Class'] == 0]
pdata = data[data['Class'] == 1]
print( ndata['Class'].value_counts())
print( pdata['Class'].value_counts())

scaler =MinMaxScaler((-1,1))
for item in data.columns:
    if data[item].max() >3 or data[item].min() <-3:
        data[[item]] = scaler.fit_transform(data[[item]])
    print( "%s, max = %f, min = %f"%(item, data[item].max(), data[item].min() ))


plt.hist( pdata['V5'] , bins = 50,edgecolor = 'blue', log = True)
plt.xlabel('V5')
plt.ylabel('Number')'''
plt.show()