import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df= quandl.get('WIKI/GOOGL')

df=df[['Adj. Open','Adj. Close','Adj. High','Adj. Low','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
forcast_C= 'Adj. Close'
forcast_Out=int(math.ceil(0.01*len(df)))
df.fillna(-9999,inplace=True)
df['label']=df[forcast_C].shift(-forcast_Out)
df.dropna(inplace=True)
X=np.array(df.drop(['label'],1))
y=np.array(df['label'])

X = preprocessing.scale(X)
y= np.array(df['label'])
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = LinearRegression()
clf.fit(X_train,y_train)

Accuracy= clf.score(X_test,y_test) 
print(Accuracy)
