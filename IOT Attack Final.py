#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


cd "C:\Users\kkpra\IOT ATTACK GENERATED DATASET\UNSW"


# In[3]:


f1 = pd.read_csv('Final_Training_Dataset.csv',low_memory=False)


# In[4]:


f1.head()


# In[5]:


f1 = f1.loc[:, ~f1.columns.str.contains('^Unnamed')]


# In[6]:


f1.head()


# In[7]:


f1.isnull().any()


# In[8]:


f1.describe()


# In[9]:


# Finding Correlation graph
cr = f1.corr()
plt.subplots(figsize=(15,10))
sb.heatmap(cr)


# In[10]:


x = cr.unstack()
c = x.sort_values()
c['Label']


# In[11]:


out = f1['Label']


# In[12]:


f1.columns


# In[13]:


# Dropping unnecessary columns
f1 = f1.drop(['Label','proto','saddr','daddr','sport','dport'],axis='columns')


# In[14]:


f1.columns


# In[15]:


# Z-score Normalisation
f1 = (f1-f1.mean())/f1.std()


# In[16]:


f1.shape


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(f1, out, test_size=0.3,random_state=42)


# ## for Testing,values of each categories graph
# one = 0
# two = 0
# three = 0
# four = 0
# for i in range(len(y_test)):
#     if y_test[i]==1:
#         one+=1
#     elif y_test[i]==2:
#         two+=1
#     elif y_test[i]==3:
#         three+=1
#     else:
#         four+=1
# print(one)
# print(two)
# print(three)
# print(four)

# # Random Forest

# In[21]:


clf1 = RandomForestClassifier(n_estimators=100, max_depth=19, random_state=0)
clf1.fit(x_train,y_train)

y_pred2=clf1.predict(x_test)

print(accuracy_score(y_test,y_pred2)*100)
conf_mat= confusion_matrix(y_test, y_pred2)
print(confusion_matrix(y_test, y_pred2))  
print(classification_report(y_test, y_pred2)) 


# In[32]:


# Saving model
import joblib
joblib.dump(clf1,'iotattack_randomForest')

modl = joblib.load('joblib_rd')


# In[33]:


#for Testing values graph
one = 0
two = 0
three = 0
four = 0
for i in tqdm(range(len(y_pred2))):
    if out[i]==1:
        one+=1
    elif out[i]==2:
        two+=1
    elif out[i]==3:
        three+=1
    else:
        four+=1
print(one)
print(two)
print(three)
print(four)


# In[19]:


conf_mat= confusion_matrix(y_test, y_pred2)
print(classification_report(y_test, y_pred2)) 
#TN
TP = []
TN = []
FP = []
FN=[]
for k in tqdm(range(len(conf_mat))):
    TP.append(conf_mat[k][k])
    sum = 0
    for i in range(len(conf_mat)):  #TN
        for j in range(len(conf_mat)):
            if i==j:
                sum = sum+conf_mat[i][i]
    TN.append(sum-TP[k])
    sum2=0
    #FP
    for i in range(len(conf_mat)):
        sum2 = sum2+conf_mat[i][k]
    FP.append(sum2-TP[k])
    #FN
    sum3=0
    for i in range(len(conf_mat)):
        sum3 = sum3+conf_mat[k][i]
    FN.append(sum3-TP[k])
    
    
print(TP)
print(TN)
print(FP)
print(FN)


for i in range(len(conf_mat)):
    print("FOR "+str(i+1)+ " class")
    false_alarm_rate = (FP[i] / float(TN[i] + FP[i]))*100
    print("False Alarm Rate: ",false_alarm_rate)
    undetected_rate = (FN[i] / float(TP[i] + FN[i]))*100
    print("Undetected Rate: " ,undetected_rate)
    


# In[35]:


#Create selector to identify features with importance more than a certain threshold
sfm = SelectFromModel(clf1)
sfm.fit(x_train, y_train)
selected_feat= x_train.columns[(sfm.get_support())]
len(selected_feat)
print(selected_feat)


# # Decision Trees

# In[36]:


from sklearn.tree import DecisionTreeClassifier

#classification via gini-index

clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=19,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=4,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, splitter='best')
#min_samples_leaf = min no of samples required for further splitting of trees
clf_gini.fit(x_train, y_train)
clf_gini
y_pred = clf_gini.predict(x_test)
y_pred
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test, y_pred))  
conf_mat= confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred)) 

#TN
TP = []
TN = []
FP = []
FN=[]
for k in range(len(conf_mat)):
    TP.append(conf_mat[k][k])
    sum = 0
    for i in range(len(conf_mat)):  #TN
        for j in range(len(conf_mat)):
            if i==j:
                sum = sum+conf_mat[i][i]
    TN.append(sum-TP[k])
    sum2=0
    #FP
    for i in range(len(conf_mat)):
        sum2 = sum2+conf_mat[i][k]
    FP.append(sum2-TP[k])
    #FN
    sum3=0
    for i in range(len(conf_mat)):
        sum3 = sum3+conf_mat[k][i]
    FN.append(sum3-TP[k])
    
    
print(TP)
print(TN)
print(FP)
print(FN)


for i in range(len(conf_mat)):
    print("FOR "+str(i+1)+ " class")
    false_alarm_rate = (FP[i] / float(TN[i] + FP[i]))*100
    print("False Alarm Rate: ",false_alarm_rate)
    undetected_rate = (FN[i] / float(TP[i] + FN[i]))*100
    print("Undetected Rate: " ,undetected_rate)


# In[37]:


clf3 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=19,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=4,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, splitter='best')
clf3.fit(x_train, y_train)
y_pred_en = clf3.predict(x_test)
y_pred_en
print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
print(confusion_matrix(y_test, y_pred))  
conf_mat= confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred)) 


#TN
TP = []
TN = []
FP = []
FN=[]
for k in range(len(conf_mat)):
    TP.append(conf_mat[k][k])
    sum = 0
    for i in range(len(conf_mat)):  #TN
        for j in range(len(conf_mat)):
            if i==j:
                sum = sum+conf_mat[i][i]
    TN.append(sum-TP[k])
    sum2=0
    #FP
    for i in range(len(conf_mat)):
        sum2 = sum2+conf_mat[i][k]
    FP.append(sum2-TP[k])
    #FN
    sum3=0
    for i in range(len(conf_mat)):
        sum3 = sum3+conf_mat[k][i]
    FN.append(sum3-TP[k])
    
    
print(TP)
print(TN)
print(FP)
print(FN)


for i in range(len(conf_mat)):
    print("FOR "+str(i+1)+ " class")
    false_alarm_rate = (FP[i] / float(TN[i] + FP[i]))*100
    print("False Alarm Rate: ",false_alarm_rate)
    undetected_rate = (FN[i] / float(TP[i] + FN[i]))*100
    print("Undetected Rate: " ,undetected_rate)
    


# In[63]:


import joblib
joblib.dump(clf3,'joblib_DT_iotattack')


# # KNN

# In[38]:


#Training and Predictions
from sklearn.neighbors import KNeighborsClassifier  
clf2 = KNeighborsClassifier(n_neighbors=5)  
clf2.fit(x_train, y_train) 

#The final step is to make predictions on our test data. To do so, execute the following script:
y_pred = clf2.predict(x_test) 


    


# In[42]:


#Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix  
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test, y_pred))  
conf_mat= confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred)) 
#TN
TP = []
TN = []
FP = []
FN=[]
for k in range(len(conf_mat)):
    TP.append(conf_mat[k][k])
    sum = 0
    for i in range(len(conf_mat)):  #TN
        for j in range(len(conf_mat)):
            if i==j:
                sum = sum+conf_mat[i][i]
    TN.append(sum-TP[k])
    sum2=0
    #FP
    for i in range(len(conf_mat)):
        sum2 = sum2+conf_mat[i][k]
    FP.append(sum2-TP[k])
    #FN
    sum3=0
    for i in range(len(conf_mat)):
        sum3 = sum3+conf_mat[k][i]
    FN.append(sum3-TP[k])
    
    
print(TP)
print(TN)
print(FP)
print(FN)

for i in range(len(conf_mat)):
    print("FOR "+str(i+1)+ " class")
    false_alarm_rate = (FP[i] / float(TN[i] + FP[i]))*100
    print("False Alarm Rate: ",false_alarm_rate)
    undetected_rate = (FN[i] / float(TP[i] + FN[i]))*100
    print("Undetected Rate: " ,undetected_rate)


# In[44]:


import joblib
joblib.dump(clf2,'joblib_knn_iotattack')


# In[ ]:





# # Stacking 3 Algorithms

# In[48]:


from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings

lr = LogisticRegression()
warnings.simplefilter('ignore')
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['Random Forest', 
                       'KNN',
                       'DT',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, f1, out, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


    


# In[62]:


conf_mat= confusion_matrix(y_test, y_pred)
    #TN
TP = []
TN = []
FP = []
FN=[]
for k in range(len(conf_mat)):
    TP.append(conf_mat[k][k])
    sum = 0
    for i in range(len(conf_mat)):  #TN
        for j in range(len(conf_mat)):
            if i==j:
                sum = sum+conf_mat[i][i]
    TN.append(sum-TP[k])
    sum2=0
    #FP
    for i in range(len(conf_mat)):
        sum2 = sum2+conf_mat[i][k]
    FP.append(sum2-TP[k])
    #FN
    sum3=0
    for i in range(len(conf_mat)):
        sum3 = sum3+conf_mat[k][i]
    FN.append(sum3-TP[k])
    
    
print(TP)
print(TN)
print(FP)
print(FN)


for i in range(len(conf_mat)):
    print("FOR "+str(i+1)+ " class")
    false_alarm_rate = (FP[i] / float(TN[i] + FP[i]))*100
    print("False Alarm Rate: ",false_alarm_rate)
    undetected_rate = (FN[i] / float(TP[i] + FN[i]))*100
    print("Undetected Rate: " ,undetected_rate)


# In[49]:


import joblib
joblib.dump(sclf,'joblib_stack_iotattack')


# In[ ]:





# # Visualising 

# ### 1- Relative Features Importance

# In[54]:





# In[66]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(x_train, y_train) 


# In[69]:


gnb_predictions = gnb.predict(x_test) 
  
# accuracy on X_test 
accuracy = gnb.score(x_test, y_test) 
print(accuracy*100)


# In[70]:


conf_mat= confusion_matrix(y_test, y_pred)
    #TN
TP = []
TN = []
FP = []
FN=[]
for k in range(len(conf_mat)):
    TP.append(conf_mat[k][k])
    sum = 0
    for i in range(len(conf_mat)):  #TN
        for j in range(len(conf_mat)):
            if i==j:
                sum = sum+conf_mat[i][i]
    TN.append(sum-TP[k])
    sum2=0
    #FP
    for i in range(len(conf_mat)):
        sum2 = sum2+conf_mat[i][k]
    FP.append(sum2-TP[k])
    #FN
    sum3=0
    for i in range(len(conf_mat)):
        sum3 = sum3+conf_mat[k][i]
    FN.append(sum3-TP[k])
    
    
print(TP)
print(TN)
print(FP)
print(FN)


for i in range(len(conf_mat)):
    print("FOR "+str(i+1)+ " class")
    false_alarm_rate = (FP[i] / float(TN[i] + FP[i]))*100
    print("False Alarm Rate: ",false_alarm_rate)
    undetected_rate = (FN[i] / float(TP[i] + FN[i]))*100
    print("Undetected Rate: " ,undetected_rate)


# In[ ]:





# In[ ]:





# In[5]:


cd C:\Users\kkpra\IOT ATTACK GENERATED DATASET\UNSW


# In[6]:


import joblib


# In[7]:


Deep = joblib.load('joblib_rd')


# In[8]:


RF = joblib.load('iotattack_randomForest')


# In[9]:


DT = joblib.load('joblib_DT_iotattack')


# In[10]:


KNN = joblib.load('joblib_knn_iotattack')


# In[11]:


Stacked = joblib.load('joblib_stack_iotattack')


# In[12]:


cd C:\Users\kkpra\IOT ATTACK GENERATED DATASET\SelfGenerated7December\datasets\final


# In[29]:


test = pd.read_csv('Testing_Dataset.csv')


# In[30]:


test.describe()


# In[31]:


test = test.iloc[:,~test.columns.str.contains('Unnamed')]


# In[32]:


test.head()


# In[33]:


test.Label.value_counts()


# In[34]:


out = test['Label']


# In[35]:


test = test.drop(['Label'],axis=1)


# In[36]:


test.columns


# In[37]:


test.columns


# In[24]:


y_pred = Deep.predict(test)


# In[27]:


y_pred


# In[46]:


len(out)


# In[48]:


#Converting one hot encoded test label to label
test = list()
for i in range(len(out)):
    for j in range(4):
        if y_pred[i][j]==1:
            test.append(j+1)


# In[52]:


test


# In[41]:


from sklearn.metrics import accuracy_score
print(accuracy_score(test,out)*100)


# In[32]:


y_pred = DT.predict(test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,out)*100)


# In[33]:


out


# In[65]:


y_pred


# In[ ]:





# # Feature importance

# In[34]:


# Calculate feature importances
importances = DT.feature_importances_


# In[38]:





# In[40]:


# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [test.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(test.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(test.shape[1]), names, rotation=90)

# Show plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


# Calculate feature importances
importances = RF.feature_importances_


# In[44]:


# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [test.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(test.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(test.shape[1]), names, rotation=90)

# Show plot
plt.show()


# In[ ]:





# In[ ]:




