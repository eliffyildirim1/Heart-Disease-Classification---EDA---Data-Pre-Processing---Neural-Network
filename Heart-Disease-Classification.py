#!/usr/bin/env python
# coding: utf-8

# ### Kalp Hastalığı Sınıflandırması - EDA - Veri Ön İşleme - Sinir Ağı

# ![Screenshot_2.png](attachment:Screenshot_2.png)

# ### Problem

# Bir hastayla ilgili klinik parametreler göz önüne alındığında, kalp hastalığı olup olmadığını tahmin edebilir miyiz?

#  Bu veri seti, kalp hastalığına sahip olma veya olmama hedef durumuyla birlikte bir dizi değişken verir. "hedef" alanı, hastada kalp hastalığının varlığını ifade eder

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("C:/Users/Elif/data/heart.csv")


# In[3]:


df.head()


# Kullanılan 14 Özellikler:
# - age - yaş
# - sex - Kişinin cinsiyeti (1 = male; 0 = female)
# - cp(chest pain type) - Yaşanan göğüs ağrısı (Değer 1: tipik anjin, Değer 2: atipik anjin, Değer 3: anjinal olmayan ağrı, Değer 4: asemptomatik)
# - trestbps - Kişinin istirahat tansiyonu
# - chol - Kişinin kolesterol ölçümü
# - fbs -  Kişinin açlık kan şekeri (1 = true; 0 = false)
# - restecg - Dinlenme elektrokardiyografik ölçümü (0 = normal, 1 = ST-T dalgası anormalliği var, 2 = Estes kriterlerine göre olası veya kesin sol ventrikül hipertrofisini gösteriyor)
# - thalach - Kişinin ulaştığı maksimum kalp atış hızı
# - exang - Egzersiz kaynaklı anjin (1 = yes; 0 = no)
# - oldpeak - Dinlenmeye göre egzersizin neden olduğu ST çökmesi
# - slope - tepe egzersiz ST segmentinin eğimi (Değer 1: yukarı eğimli, Değer 2: düz, Değer 3: aşağı eğimli)
# - ca - florosopi ile renklendirilen ana damarların sayısı
# - thal - Talasemi adı verilen bir kan hastalığı (3 = normal; 6 = sabit kusur; 7 = geri döndürülebilir kusur)
# - target - Kalp hastalığı (1=yes, 0=no)

# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


plt.figure(figsize=(14,10))
sns.heatmap(df.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()


# ### KEŞİFSEL VERİ ANALİZİ

# In[8]:


df.target.value_counts()


# In[9]:


sns.countplot(x="target", data=df, palette="bwr")
plt.show()


# Kalp hastası sayısının daha çoğunlukta olduğunu görüyoruz.

# In[10]:


sns.distplot(df['target'],rug=True)
plt.show()


# In[11]:


countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


# Veri setindeki kişilerin kalp hastası olmayanların %45 ve üzeri, kalp hastası olanların ise %54 ve üzeri dağılımı olduğunu söyleyebiliriz.

# In[12]:


sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# Kalp hastası olan kişilere baktığımızda erkeklerin kalp hastası olma riskinin daha fazla olduğunu söyleyebiliriz.

# In[13]:


countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))


# Kadın hasta yüzdesi %31 e karşılık gelirken Erkek hasta yüzdesi %68 e denk gelmektedir.

# In[14]:


df.groupby('target').mean()


# In[15]:


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# Yaşlara Göre Kalp Hastalığı Sıklığına baktığımızda 42- 55 yaş aralığında kalp hasta olma riski fazla olduğunu ve 61 yaşın kalp hastası olma riskinin daha az olduğunu görüyoruz. 

# Parametrelerin Yaşa Göre Kalp Hastalığına Etkisi

# In[16]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.scatter(x=df.age[df.target==1],y=df.thalach[df.target==1],c='blue')
plt.scatter(x=df.age[df.target==0],y=df.thalach[df.target==0],c='black')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend(['Disease','No Disease'])

plt.subplot(2,2,2)
plt.scatter(x=df.age[df.target==1],y=df.chol[df.target==1],c='red')
plt.scatter(x=df.age[df.target==0],y=df.chol[df.target==0],c='green')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend(['Disease','No Disease'])

plt.subplot(2,2,3)
plt.scatter(x=df.age[df.target==1],y=df.trestbps[df.target==1],c='cyan')
plt.scatter(x=df.age[df.target==0],y=df.trestbps[df.target==0],c='fuchsia')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.legend(['Disease','No Disease'])

plt.subplot(2,2,4)
plt.scatter(x=df.age[df.target==1],y=df.oldpeak[df.target==1],c='grey')
plt.scatter(x=df.age[df.target==0],y=df.oldpeak[df.target==0],c='navy')
plt.xlabel('Age')
plt.ylabel('ST depression')
plt.legend(['Disease','No Disease'])
plt.show()


# Cinsiyete Göre Kalp Hastalığı Sıklığı

# In[17]:


pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[18]:


plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# Görünüşe göre biri ne kadar gençse, maksimum kalp atış hızı o kadar yüksek (noktalar grafiğin solunda daha yüksektir) ve kişi ne kadar yaşlıysa, o kadar çok mavi nokta vardır. Ancak bunun nedeni grafiğin sağ tarafında hep birlikte daha fazla nokta bulunması olabilir (daha yaşlı katılımcılar).

# Kişinin Açlık Kan Şekerine Göre Kalp Hastalığı Sıklığı 

# In[19]:


pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()


# Yaşanan Göğüs Ağrısına Göre Kalp Hastalığı Sıklığı  

# In[20]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()


# Değer 1: tipik anjin, Değer 2: atipik anjin, Değer 3: anjinal olmayan ağrı

# In[21]:


pd.crosstab(df['target'], df['ca']).plot(kind="bar", figsize=(10,6))

plt.title("Target distribution for ca")
plt.xlabel("0 = No Heart Disease, 1 = Heart Disease")
plt.ylabel("Count")
plt.legend(["0", "1", "2", "3", "4"])
plt.xticks(rotation=0);


# Creating Dummy Variables

# In[22]:


a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")


# In[23]:


frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()


# In[24]:


df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()


# ### Keras Neural Network

# In[25]:


X = df.drop(['target'], axis = 1)
y = df.target.values


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:





# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[28]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings


# ### ANN

# In[29]:


model = Sequential()

input_number = X_train.shape[1] 

model = Sequential()

model.add(Dense(20, activation='relu', input_shape=(input_number,), name = "Hidden_Layer_1"))
model.add(Dense(11, activation='relu', name = "Hidden_Layer_2"))
model.add(Dense(1, name = "Output"))
model.summary()


# In[30]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[31]:


model.fit(X_train, y_train, epochs=10)


# In[32]:


y_pred = model.predict(X_test)


# In[33]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)
#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred.round())
print('accuracy of the model: ',ac)


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
rdf_ac=accuracy_score(rdf_pred,y_test)
plt.title("rdf_cm")
sns.heatmap(rdf_cm,annot=True,fmt="d",cbar=False)
print('RandomForest_accuracy:',rdf_ac)


# In[35]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
def plotting(true,pred):
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    precision,recall,threshold = precision_recall_curve(true,pred[:,1])
    ax[0].plot(recall,precision,'g--')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))
    fpr,tpr,threshold = roc_curve(true,pred[:,1])
    ax[1].plot(fpr,tpr)
    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))
    ax[1].plot([0,1],[0,1],'k--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
plotting(y_test,rdf_c.predict_proba(X_test))
plt.figure()   
    


# In[ ]:




