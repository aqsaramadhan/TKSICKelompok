from sys import displayhook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("D:\Heart\csv\heart_failure_clinical_records_dataset.csv")
df.head() ##this returns the first five rows of the dataset

df.shape ##returns the no. of rows and columns
df.dtypes
df.describe
df.isnull().sum() ##check for null values
df.duplicated().any() ##check for duplicate values

df['DEATH_EVENT'].value_counts().plot(kind='bar')
plt.show()

fig,ax = plt.subplots(1,2,figsize=(16,8))
ax[0].hist(df['age'],label = 'patients')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Number of Patients')
ax[0].set_yticks([5,10,15,20,25,30,35,40,45,50,55,60])
ax[0].legend()
ax[0].set_title('Age Distribution')
ax[1].hist(x = [df[df['DEATH_EVENT']==1]['age'],df[df['DEATH_EVENT']==0]['age']],stacked=True,label=['Dead','Survived'])
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Number of patients')
ax[1].set_yticks([5,10,15,20,25,30,35,40,45,50,55,60])
ax[1].set_title('Distribution of age against Death_event')
ax[1].legend()

fig,ax = plt.subplots(1,2,figsize=(16,8))
ax[0].hist(df['serum_creatinine'], label = 'patients')
ax[0].set_xlabel('serum_creatinine')
ax[0].set_ylabel('Number of Patients')
ax[0].set_yticks([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
ax[0].legend()
ax[0].set_title('serum_creatinine Distribution')
ax[1].hist(x = [df[df['DEATH_EVENT']==1]['serum_creatinine'],df[df['DEATH_EVENT']==0]['serum_creatinine']], stacked=True, label=['Dead','Survived'])
ax[1].set_xlabel('serum_creatinine')
ax[1].set_ylabel('Number of patients')
ax[1].set_yticks(([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220]))
ax[1].set_title('Distribution of serum_creatinine against Death_event')
ax[1].legend()

fig,ax = plt.subplots(1,2,figsize=(20,10))
ax[0].hist(df['serum_sodium'], label = 'patients')
ax[0].set_xlabel('serum_sodium')
ax[0].set_ylabel('Number of Patients')
ax[0].set_yticks([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
ax[0].legend()
ax[0].set_title('Serum sodium Distribution')
ax[1].hist(x = [df[df['DEATH_EVENT']==1]['serum_sodium'],df[df['DEATH_EVENT']==0]['serum_sodium']], stacked=True, label=['Dead','Survived'])
ax[1].set_xlabel('serum_sodium')
ax[1].set_ylabel('Number of patients')
ax[1].set_yticks([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
ax[1].set_title('Distribution of serum_sodium against Death_event')
ax[1].legend()

plt.figure(figsize=[16,8])
corr = sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")

features = df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']]
X = df[['age', 'ejection_fraction', 'serum_sodium', 'serum_creatinine']]
y = df['DEATH_EVENT']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    displayhook(X_test)

from sklearn import svm
svm = svm.SVC()
svm.fit(x_train, y_train)
predictions = svm.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print("Confusion Matrix : \n\n" , confusion_matrix(predictions,y_test))
print("Classification Report : \n\n" , classification_report(predictions,y_test),"\n")

import pickle
pickle.dump(svm, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model)

    