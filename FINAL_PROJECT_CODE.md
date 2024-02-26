FINAL PROJECT CODE

IMPORTS AND DIRECTORY


```python
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import sklearn as sk

from datetime import datetime
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

from numpy.random import seed
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf            # main ANN librray
from tensorflow import keras
from tensorflow import random
from tensorflow.keras import layers
from keras.models import Sequential
#from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers.experimental import preprocessing


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error 
from mlxtend.plotting import plot_decision_regions
```


```python
os.chdir('C:\\Users\\Owner\\Documents\\DSC 325\\Final Project')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[2], line 1
    ----> 1 os.chdir('C:\\Users\\Owner\\Documents\\DSC 325\\Final Project')
    

    FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Owner\\Documents\\DSC 325\\Final Project'


DATA CLEANING


```python
# Loading data
df = pd.read_csv('FINALPROJECT.csv')
print('Shape of DF:', df.shape)
df.tail(4)
```


```python
#Drop null values
df = df.dropna()
print('Shape of DF:', df.shape)
```


```python
df.info()
```


```python
df["nameOrig"] = df["nameOrig"].astype('category')
df["type"] = df["type"].astype('category')
df["nameDest"] = df["nameDest"].astype('category')
df.info()
```


```python
df.drop_duplicates(subset=None, keep="first", inplace=True)
df.info()
```


```python
p1 = df[df['isFraud'] == 1]
p1sps=p1.groupby('isFraud').apply(lambda x: x.sample(frac=0.87565674))
p2 = df[df['isFraud'] == 0]
p2sps=p2.groupby('isFraud').apply(lambda x: x.sample(frac=0.00095472))
df_mod = pd.concat([p1sps, p2sps], axis=0)
df_mod
```


```python

```

EDA


```python
srs = df.sample(frac=0.01, random_state=123)
```


```python
srs
```


```python
#EDA of Response isFraud
tb1 = df['isFraud'].value_counts() # value count
print(tb1) # freq table
print(tb1/sum(tb1)) # perc table

```


```python
tb2 = df_mod['isFraud'].value_counts() # value count
print(tb2) # freq table
print(tb2/sum(tb2)) # perc table

```


```python
amnt= sns.boxplot(y="amount", x= "isFraud" ,  data=df_mod)
amnt.set(xlabel='Is Fraud', ylabel='Amount transferred (hundred-thousands)')

```


```python
OldBal = sns.boxplot(y="oldbalanceDest", x= "isFraud", data=df_mod)
OldBal.set(xlabel='Is Fraud', ylabel='Beginning Balance of the Destination account')

```


```python
NewBal = sns.boxplot(y="newbalanceOrig", x= "isFraud", data=df_mod)
NewBal.set(xlabel='Is Fraud', ylabel='Ending Balance of the Origin account')

```


```python
df.groupby('isFraud')['oldbalanceDest'].median()
```


```python
df.groupby('isFraud')['newbalanceDest'].median()
```


```python
df.groupby('isFraud')['amount'].median()
```


```python
df.groupby('isFraud')['oldbalanceOrg'].median()
```


```python
df.groupby('isFraud')['newbalanceOrig'].median()
```


```python
 sns.boxplot(y="oldbalanceOrg", x= "isFraud", data=data_mod)
```


```python
#Crosstable 
tb = df_mod['type'].value_counts(sort=True, normalize=False)
print(tb)                          
print(tb/sum(tb))                  
pd.crosstab(index=df_mod['type'], columns=df_mod['isFraud'])        
```


```python
X = df_mod[["amount","oldbalanceOrg", "newbalanceOrig","oldbalanceDest"]]
y = df_mod["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=0)
print('Train, Test set shape:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```


```python
# Scaling predictors

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```


```python
# ANN model structure 
# units: number of units in hidden layer
# input_shape: shape of the input var(s)

model = Sequential()
model.add(layers.Dense(units=6, activation="relu"))
model.add(layers.Dense(units=6, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))

```


```python
%%time
# model compile, fit, 17.9 seconds for 100 epochs on CPU for 1 x 2 x 1 ANN
# loss function options: binary_crossentropy
# optimizer: adam is an extension to stochastic gradient descent
# metrics: accuracy 

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

history = model.fit(
    X_train_std, y_train,           # scaled train predictor, train response
    epochs = 100,                  # total no. of feeding the entire training data into network
    verbose = 0,                    # 0: no show, 1: details, 2: summary
    batch_size = 16,                # cases are fed in batches of batch_size
    validation_split = 0.1)         # % of train set used for perfromance evaluation
```


```python
# Defining a function for plotting accuracy in train & validation sets

def plot_loss(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)
```


```python
# printing model performance on train & validation, test set

print('Train accuracy on last epoch:', history.history['accuracy'][-1])             # last accuracy on tarin set
print('Validation accuracy on last epoch:', history.history['val_accuracy'][-1])    # last accuracy on validation set
print('Test accuracy:', model.evaluate(X_test_std, y_test, verbose=0))         # accuracy on test set
```


```python
print(model.predict(sc.transform([[181, 181, 0, 21282, 0]])) > 0.5)

```


```python
#prediction with median values when isFraud = 1
print(model.predict(sc.transform([[353179.45,348705.145,0,0]])))
```


```python
#prediction with median values when isFraud = 0 
print(model.predict(sc.transform([[76214.97,15937,0,126863.76]])))
```


```python
model.save('Finalmodel.h5')

```


```python

```
