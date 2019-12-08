import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("Dataset/creditcard.csv")
data = data.dropna(axis =0)
X = pd.DataFrame(data.loc[:, 'Time': 'Amount'])
y = data['Class']

#scaling of features
scaler =MinMaxScaler((-1,1))
X[X.columns] = scaler.fit_transform(X[X.columns])

yp = y[y==1]
Xp = X[y==1]
yn = y[y==0]
Xn = X[y==0]
Xtrain, Xtest, ytrain, ytest = train_test_split(Xn,yn,test_size = 0.3)
Xptrain, Xptest, yptrain, yptest = train_test_split(Xp,yp,test_size = 0.3)

Xtrain = pd.concat([Xtrain, Xptrain])
Xtest = pd.concat([Xtest, Xptest])
ytrain = pd.concat([ytrain, yptrain])
ytest = pd.concat([ytest, yptest])

# precision, recall and f1-score for the 1.0 case are important
model = RandomForestClassifier( n_estimators = 30,
                              bootstrap = True,
                              max_features = 'sqrt')
model.fit(Xtrain,ytrain)
yrfor = model.predict( Xtest)
matrix = classification_report ( yrfor, ytest)
print( matrix)

featureimp = pd.DataFrame({'feature':list(X.columns),
                           'importance':100*model.feature_importances_}).\
                            sort_values('importance',ascending = False)