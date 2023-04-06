#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd


# In[84]:


housing = pd.read_excel("E:/code with harry/real_estate_housing_data.xlsx")


# In[85]:


housing.head()


# In[86]:


housing.info()


# In[87]:


# chas column is categorical column
# we will find value counts
housing['CHAS'].value_counts()


# In[88]:


housing.describe()


# In[89]:


import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


housing.hist(bins=50, figsize=(20,15))


# In[91]:


#mdev,lstat,dis are right_skewed
# Age is left skewed
#RM is close to normal distribution
# rest are not following any famous distribution


# In[92]:


#import numpy as np
#ef split_train_test(data,test_ratio):
  #  np.random.seed(42)
   # shuffled = np.random.permutation(len(data))
    #test_set_size = int(len(data)*test_ratio)
    #test_indices = shuffled[:test_set_size]
    #train_indices = shuffled[test_set_size:]
    #return data.iloc[train_indices], data.iloc[test_indices]


# In[93]:


#train_set, test_set = split_train_test(housing, 0.2)


# In[94]:


#print (len(train_set), len(test_set))


# In[95]:


# Splitting data into train, test
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)
print (len(train_set), len(test_set))


# In[96]:


# here, CHAS column is categorical column of 1 and 0. chances is that may be 1 does not come in our test set so we have to divide
# 1 and 0 in equl ration in train, test so we use stratfiedShuffleSplit method of sklearn
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]


# In[97]:


start_test_set['CHAS'].value_counts()


# In[98]:


start_train_set['CHAS'].value_counts()


# # Looking for correlation

# In[99]:


# Now we will use pearson corr to find correlation between MEDV output column with other column
corr_matrix = housing.corr()


# In[100]:


corr_matrix["MEDV"].sort_values(ascending=False)


# In[101]:


# we will now draw plot to look the relation of everey column with each other.
#after getting plot we will find out which column have high correlation and a outlier in them
from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'LSTAT', 'ZN']
scatter_matrix(housing[attributes], figsize = (12,8))


# In[102]:


housing.plot(kind="scatter", x="RM", y="MEDV")


# The plot show us there is a strong positive linear relationship btn RM and MEDV.
# There is few outlier in the data. 
# There is cap on 50 that means price will not go above 50.

# In[ ]:





# ## Using Feature Engineering
# try to find new column

# In[103]:


housing["TAXRM"] = housing["TAX"]/housing["RM"]


# In[ ]:





# In[104]:


corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# TAXRM have highly negative co-relation

# In[105]:


housing = start_train_set.drop("MEDV", axis =1)
housing_label =start_train_set['MEDV'].copy()


# # Creating PipeLine

# ## Feature Scaling
# 
# Feature scaling are of two types: 1)Min_max scaling(Normalization) 2) Standardization
# 1)min-max = value-min/max-min. sk learn provide a class called MinMaxScaler
# 2) standardization means (value-mean)/std. sklearn provide class called standarScaler

# In[106]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])


# In[107]:


housing_num = my_pipeline.fit_transform(housing)


# In[108]:


housing_num.shape
# it is a numpy array


# ## Selecting desire model
# 

# In[109]:


# Splitting data into train, test
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_num, test_size = 0.2, random_state=42)
print (len(train_set), len(test_set))


# In[110]:


housing_num.shape


# In[111]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#model = DecisionTreeRegressor()
#model = LinearRegression()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num, housing_label)


# In[112]:


some_data = housing.iloc[:5]


# In[113]:


some_label = housing_label.iloc[:5]


# In[114]:


prepared_data = my_pipeline.transform(some_data)


# In[115]:


model.predict(prepared_data)


# In[116]:


list(some_label)


# In[117]:


from sklearn.metrics import mean_squared_error
import numpy as np
housing_prediction = model.predict(housing_num)
mse = mean_squared_error(housing_label, housing_prediction)
rmse = np.sqrt(mse)


# In[118]:


mse


# In[119]:


rmse


# Our model is doing overfitting while descision tree
# and in linear regression error is high
# 

# # To over come overfitting we have to choose better model
# We will choose Cross validation

# In[120]:


from sklearn.model_selection import cross_val_score
import numpy as np
score = cross_val_score(model, housing_num, housing_label, scoring="neg_mean_squared_error", cv =10)
rmse_score = np.sqrt(-score)


# In[121]:


rmse_score


# In[122]:


def print_scores(score):
    print("score:", score)
    print("Mean:", score.mean())
    print("Standard_Deviation:",score.std())


# In[123]:


print_scores(rmse_score)


# # 1)DescisionTree OutPut
#    Mean: 4.37641635272032
#    Standard_Deviation: 1.1007453263565665
#    
#  2) Linear Regression
#     Mean: 4.318838211798885
#     Standard_Deviation: 1.2239657125417343
# 
# 3) random_forest_regressor
#     Mean: 3.2891837634299828
#     Standard_Deviation: 0.7314230402463477
#     
# In all 3 model random forest is doing least error there we will choose random forest

# # Saving the model

# In[124]:


from joblib import dump,load
dump(model, 'real_estate.joblib')


# # Testing The Model

# In[126]:


X_test = start_test_set.drop("MEDV", axis=1)
Y_test = start_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_prediction = model.predict(X_test_prepared)
final_mse= mean_squared_error( Y_test, final_prediction)
final_rmse = np.sqrt(final_mse)


# In[127]:


final_rmse


# In[129]:


print(final_prediction, list(Y_test))


# In[131]:


prepared_data[0]


# # Model usage

# In[132]:


from joblib import dump,load
import numpy as np
model = load('real_estate.joblib')
features = np.array([[-3.43942006,  10.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  9.61111401, -7.0016859 , -0.5778192 ,
       -0.97491834,  20.41164221, -9.86091034]])
model.predict(features)


# In[ ]:




