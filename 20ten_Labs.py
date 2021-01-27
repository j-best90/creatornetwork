#!/usr/bin/env python
# coding: utf-8

# ## Import Primary Libraries

# In[188]:


## Import Primary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

## Import & Split Sheets into Dataframes
xls = pd.ExcelFile("VypeVuse_Creator_Network_ModelTest.xlsx")
df_details = pd.read_excel(xls, 'Creator_Details')
df_signup = pd.read_excel(xls, 'Creator_Network_Sign_Up')
df_perf_context = pd.read_excel(xls, 'Creator_Performance_Context')
df_statistics = pd.read_excel(xls, 'Creator_Social_Stats')
df_YT_data = pd.read_excel(xls, 'YouTube_Data')
df_lookup = pd.read_excel(xls, 'Creator_Lookup')


# ## Import & Split Sheets into Dataframes

# In[189]:


xls = pd.ExcelFile("VypeVuse_Creator_Network_ModelTest.xlsx")
df_details = pd.read_excel(xls, 'Creator_Details')
df_signup = pd.read_excel(xls, 'Creator_Network_Sign_Up')
df_perf_context = pd.read_excel(xls, 'Creator_Performance_Context')
df_statistics = pd.read_excel(xls, 'Creator_Social_Stats')
df_YT_data = pd.read_excel(xls, 'YouTube_Data')
df_lookup = pd.read_excel(xls, 'Creator_Lookup')


# ## Details Data Preprocessing

# In[190]:


df_details["Primary_Content_Category"] = df_details["Primary_Content_Category"].astype('category')
df_details["Primary_Content_Category_Index"] = df_details["Primary_Content_Category"].cat.codes
df_details["Total_Audience"] = df_details["IG_Audience"] + df_details["FB_Audience"] + df_details["TW_Audience"] + df_details["YT_Audience"]
df_details_mdl = df_details[["Creator_ID","Primary_Content_Category_Index","Total_Audience"]]
df_details_mdl.head()


# ## Sign Up Data Preprocessing

# In[191]:


df_signup.head()


# ## Performance Context Preprocessing

# In[192]:


df_perf_context.head()


# ## Social Statistcs Preprocessing

# In[193]:


df_statistcs_mdl = df_statistics[["Creator_ID","Reach_per_Post","Perc_Audience_Reached_Feed"]]


# ## YouTube Data Preprocessing

# In[194]:


df_YT_data_mdl = df_YT_data[["Creator_ID","YouTube_Video_Views"]]


# ## Merging Datasets & Preparing Data for Modeling

# In[195]:


#Merge
Model_v1 = pd.merge(df_details_mdl, df_statistcs_mdl, how='left', on='Creator_ID', left_index=True, sort=True,copy=False)
Model_v2 = pd.merge(Model_v1, df_YT_data_mdl, how='left', on='Creator_ID', left_index=True, sort=True,copy=False)

#Fill Null Values 
#values = {'YouTube_Video_Views': 0}
Model_v2['YouTube_Video_Views'].fillna(0, inplace = True)

Model_v3 = Model_v2[Model_v2.YouTube_Video_Views != 0]


# ## Identifying Independent and Target Variables

# In[196]:


ind = Model_v2[['Primary_Content_Category_Index','Total_Audience','Reach_per_Post','Perc_Audience_Reached_Feed']].reset_index()
ind = ind[['Primary_Content_Category_Index','Total_Audience','Reach_per_Post','Perc_Audience_Reached_Feed']]
target = Model_v2[['YouTube_Video_Views']].reset_index()
target = target[['YouTube_Video_Views']]


# In[197]:


train_df = Model_v3[['YouTube_Video_Views','Primary_Content_Category_Index','Total_Audience','Reach_per_Post',
                  'Perc_Audience_Reached_Feed']]

test_df = Model_v2[['Creator_ID','Primary_Content_Category_Index','Total_Audience','Reach_per_Post',
                  'Perc_Audience_Reached_Feed']]


# In[225]:


X_train = train_df.drop("YouTube_Video_Views", axis=1)
X_train1 = train_df['Total_Audience']
Y_train = train_df["YouTube_Video_Views"]
X_test  = test_df.drop("Creator_ID", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# ___

# ## Support Vector Machine

# In[201]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ## K-Nearest Neighbours

# In[202]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ## Gaussian Naive Bayes

# In[203]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ## Multiple Linear Regression

# In[206]:


regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
regr_y_hat = regr.predict(X_test)
acc_regr = round(regr.score(X_train, Y_train) * 100, 2)
acc_regr


# ## Simple Linear Regression

# In[ ]:





# ## Evaluating the Models

# In[216]:


models = pd.DataFrame({
    'Model': ['Multiple Linear Regression','Gaussian Naive Bayes','K-Nearest Neighbours'],
    'Score': [acc_regr,acc_gaussian,acc_knn]})

models.sort_values(by='Score', ascending=False)


# ## Creating a New Data Frame with Predictions Added from our Preferred Model

# In[213]:


ML_dataframe = pd.DataFrame({
        "Creator_ID": test_df["Creator_ID"],
        "Est. YT Views": regr_y_hat.astype('int')
    })

ML_dataframe1 = pd.DataFrame({
        "Creator_ID": test_df["Creator_ID"],
        "Est. YT Views": Y_pred.astype('int')
    })

ML_dataframe.to_csv('ML_dataframe.csv', index=False)

ML_dataframe1.to_csv('ML_dataframe1.csv', index=False)


# ## Merging Predicted Values with Original Data

# In[229]:


Final_DF = pd.merge(Model_v2, ML_dataframe1, how='left', on='Creator_ID', left_index=True, sort=True,copy=False)


# In[230]:


Final_DF.set_index('Creator_ID')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Creating Train & Test Split

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[148]:


x_train, x_test, y_train, y_test = train_test_split(ind, target,test_size=0.3)


# ## Training the Model

# In[149]:


creatorTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

creatorTree.fit(x_train,y_train)


# In[152]:


creatorPredTree = creatorTree.predict(x_test)
print (creatorPredTree [0:5])
print (y_test [0:5])


# In[154]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, creatorPredTree))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




