#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("loan.csv")


# In[3]:


df.shape


# # Introduction To The Data:
# The csv data file is provided with the project in order to refer to it later in our data exploration. This contains information about the various columns and will be useful when we clean up the dataset.

# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe()


# Another notable thing to remove is to remove columns with more than 70% missing values. It would be time consumming and inefficient to deal with the tremendous amount of missing values from these columns. so remove all the columns with more than 70% missing data as they won't be helping for modelling and exploration.

# In[8]:


temp = [i for i in df.count()<163987 *0.30]
df.drop(df.columns[temp],axis=1,inplace=True)


# In[11]:


# half_point = len(df) / 3
# df = df.dropna(thresh=half_point, axis=1)
# # we save the new file
# df.to_csv('loan.csv', index=False)


# In[12]:


df = pd.read_csv('loan.csv', low_memory = False)
df.drop_duplicates()

df.iloc[0]


# As we can see, the number of columns will be something to work on. We will remove the addr_state column mostly because it leads to or describes information that is not necessary for our analysis.

# # Variable	Description
# 
# ![image.png](attachment:image.png)

# We now have 14 columns to work with. We removed 1 column of unuseful information and this will make the data easier to process and fit with the machine learning algorithm. But we are not done.

# # Target Column
# The target column is a critical part when fitting this type of data to machine learning algorithms because it tries to make prediction based on the outcome that we want. In this particular case, we want to predict the bad loan (bad_loan) which can take values (2) in total.

# In[13]:


df['bad_loan'].value_counts()


# In[15]:


df['bad_loan'].value_counts().plot(kind= 'barh', color = 'yellow', title = 'Possible bad loan', alpha = 0.75)
plt.show()


# only 2 values are important in our model's binary classification; 1 and 0. These 2 values indicate the result of the loan outcome.

# In[16]:


plt.figure(figsize = (9,5))
df['bad_loan'].plot(kind ='hist')


# In[17]:


sns.distplot (df['bad_loan'])
fig=plt.figure()


# In[18]:


df['term'].value_counts(normalize=True).plot.bar()


# In[19]:


df['home_ownership'].value_counts(normalize=True).plot.bar()


# In[20]:


df['purpose'].value_counts(normalize=True).plot.bar()


# In[21]:


df['addr_state'].value_counts(normalize=True).plot.bar()


# In[22]:


df['verification_status'].value_counts(normalize=True).plot.bar()


# In[28]:


term=pd.crosstab(df['term'],df['bad_loan']) 
term.div(term.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[29]:


emp_length=pd.crosstab(df['emp_length'],df['bad_loan']) 
emp_length.div(emp_length.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[30]:


home_ownership=pd.crosstab(df['home_ownership'],df['bad_loan']) 
home_ownership.div(home_ownership.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[31]:


purpose=pd.crosstab(df['purpose'],df['bad_loan']) 
purpose.div(purpose.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[32]:


# addr_state=pd.crosstab(df['addr_state'],df['bad_loan']) 
# addr_state.div(addr_state.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[33]:


delinq_2yrs=pd.crosstab(df['delinq_2yrs'],df['bad_loan']) 
delinq_2yrs.div(delinq_2yrs.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[34]:


# longest_credit_length=pd.crosstab(df['longest_credit_length'],df['bad_loan']) 
# longest_credit_length.div(longest_credit_length.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[35]:


verification_status=pd.crosstab(df['verification_status'],df['bad_loan']) 
verification_status.div(verification_status.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,6),color = ['r','g'])


# In[39]:


matrix = df.corr() 
f, ax = plt.subplots(figsize=(15, 10)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")


# # Outlier Treatment - Through Log Transformation

# In[40]:


df['loan_amnt_log'] = np.log(df['loan_amnt'])
df['loan_amnt_log'].hist(bins=25)


# # Final Data Cleaning
# Single value columns are not information that help our model, it does not provide any insight into the likelihood of default or repayment, and will be removed from the dataset.

# In[41]:


df.shape


# # Preparing The Features: Dealing With Missing Values
# We start with the filtered loan_data.csv from the previous analysis. Do we have many Null values in the file loan_data? We will look at how we can handle these values. We need to deal with non-numeric value and null values, because scikit-learn assume that the values are numeric and filled, otherwise it could throw an error or miss evaluate the data.

# In[42]:


null_counts = df.isnull().sum()
null_counts


# We have 6 columns with missing values: 29 with longest_credit_length, 29 with total_acc, 193 with revol_util, 29 with delinq_2yrs and 5804 with emp_lenght. Instead of removing the columns, we will remove the rows. We consider rows as incomplete (in a real life setting, we would reject the application simply because it is not complete).

# In[43]:


df = df.drop("emp_length", axis=1)
df = df.dropna(axis=0)


# In[44]:


df.shape


# # Handling Non-Numeric Data Types
# The data types of columns are important to look at and we will need to deal with non-numeric values in order to encode and use them in our machine learning algorithms.

# In[45]:


print(df.dtypes.value_counts())


# In[46]:


# Number of each type of column
df.dtypes.value_counts().sort_values().plot(kind='barh')
plt.title('Number of columns distributed by Data Types',fontsize=20)
plt.xlabel('Number of columns',fontsize=15)
plt.ylabel('Data type',fontsize=15)


# We have 5 objects that need to be addressed.

# In[47]:


object_columns_df = df.select_dtypes(include=["object"])
print(object_columns_df.iloc[0])


# In[48]:


df['term'].value_counts()


# In[49]:


df['home_ownership'].value_counts()


# In[50]:


df['purpose'].value_counts()


# In[51]:


df['addr_state'].value_counts()


# We will remove the addr_state because if we were to encode all these variables, we would make our dataframe quite large and would slow the computation done by our machine learning algorithm.

# In[52]:


df['verification_status'].value_counts()


# In[53]:


cleanup_df = {'term':{'36 months':2, '60 months':4},
              'home_ownership':{'MORTGAGE':12, 'RENT':10, 'OWN':8, 'OTHER':6, 'NONE':4, 'ANY':2},
              'purpose':{'debt_consolidation':26, 'credit_card':24, 'other':22, 'home_improvement':20, 'major_purchase':18,         
              'small_business':16, 'car':14, 'medical':12, 'wedding':10, 'moving':8, 'house':6, 'vacation':4, 'educational':2, 'renewable_energy':0},
              'verification_status':{'verified':2, 'not verified':0}
             }


# In[54]:


df.replace(cleanup_df, inplace=True)
df.head()


# In[55]:


# df = df.drop(["addr_state"],axis=1)


# In[56]:


df.head()


# In[57]:


from sklearn.model_selection import train_test_split

x = df.drop(['bad_loan',"addr_state"] , axis=1)
# print(x_train)

y = df["bad_loan"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
# print(x_test)
#print (x_train.shape)

x_train.head()


# In[59]:


print(x_test)
print(x_train)


# # KNeighborsClassifier

# In[60]:


import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print ("K-NN Classification Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set    : {:.2f}'.format(knn.score(x_test, y_test)))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[61]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()


# # Decision Tree

# In[62]:


from sklearn import tree

dtree = tree.DecisionTreeClassifier(random_state=2)
dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)
print (y_pred)
print ("Decision Tree Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(dtree.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtree.score(x_test, y_test)))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # K-fold Cross Validation

# In[63]:


#K-fold Cross Validation for K-NN

from sklearn.model_selection import cross_val_score
import numpy as np

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=15)

#train model with cv of 15 
cv_scores = cross_val_score(knn_cv, x, y, cv=15)

#print each cv score (accuracy) and average them
print(cv_scores)
print("Mean Accuracy: ",np.mean(cv_scores))
print("Max Accuracy: ",np.max(cv_scores))


# In[64]:


#K-fold Cross Validation for Decision Tree

from sklearn.model_selection import cross_val_score
import numpy as np

#create a new DT model
DT_cv = tree.DecisionTreeClassifier(random_state=1)

#train model with cv of 15 
cv_scores = cross_val_score(DT_cv, x, y, cv=15)

#print each cv score (accuracy) and average them
print(cv_scores)
print("Mean Accuracy: ",np.mean(cv_scores))
print("Max Accuracy: ",np.max(cv_scores))


# # Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print("cv: ", confusion_matrix(y_test, y_pred))
print(" Accuracy: ",accuracy_score(y_test, y_pred))
print("Accuracy on Logistic Regression training set: {:.2f}".format(logreg.score(x_train, y_train)))
print("Accuracy on Logistic Regression test set: {:.2f}".format(logreg.score(x_test, y_test)))


# In[ ]:


# from sklearn.svm import SVC

# model=SVC()
# model.fit(x_train,y_train)
# pred=model.predict(x_test)

# print ("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


# In[ ]:


# #K-fold Cross Validation for SVM

# from sklearn.model_selection import cross_val_score
# import numpy as np

# #create a new SVM model

# svm_cv = SVC()

# #train model with cv of 15 
# cv_scores = cross_val_score(svm_cv, x, y, cv=15)

# #print each cv score (accuracy) and average them
# print(cv_scores)
# print("Mean Accuracy: ",np.mean(cv_scores))
# print("Max Accuracy: ",np.max(cv_scores))

