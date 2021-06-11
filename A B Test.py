#!/usr/bin/env python
# coding: utf-8

# You are an analyst at a big online store. Together with the marketing department, you've compiled a list of hypotheses that may help boost revenue.
# You need to prioritize these hypotheses, launch an A/B test, and analyze the results.

# In[1]:


import pandas as pd

import datetime as dt
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


import sys
import warnings
if not sys.warnoptions:
       warnings.simplefilter("ignore")


# ## Part 1. Prioritizing Hypotheses

# To priotize problems there is 'ICE' method which is also modified as 'RICE'. These framework are name after components used to priotize 
# hypothesis. Components used in these frameworks are:
#     
# * _Reach_ — user reach, on a scale of one to ten
# 
# * _Impact_  — impact on users, on a scale of one to ten
# 
# * _Confidence_  — confidence in the hypothesis, on a scale of one to ten
# 
# * _Effort_ — the resources required to test a hypothesis, on a scale of one to ten. The higher the Effort value, the more resource-intensive the test.
# 

#  ICE= Impact*Confidence/ Effort
#     
#  RICE= Reach*Impact*Confidence/ Effort

# In[3]:


pd.set_option('max_colwidth', 400)


# In[4]:


hypothesis= pd.read_csv('/datasets/hypotheses_us.csv', sep=';')


# In[5]:


print(hypothesis.head())


# In[6]:


hypothesis['ICE']= (hypothesis['Impact']*hypothesis['Confidence']/hypothesis['Effort']).round(3)


# In[7]:


ice= hypothesis.sort_values('ICE', ascending=False)


# In[8]:


ice


# In[9]:


hypothesis['RICE']= (hypothesis['Reach']*hypothesis['Impact']*hypothesis['Confidence']/hypothesis['Effort']).round(3)


# In[10]:


rice= hypothesis.sort_values('RICE', ascending=False)


# In[11]:


rice


# In[45]:


#Make x and y variables for success rate data
x = hypothesis['RICE'].values
y = hypothesis['ICE'].values
#types = hypothesis.index()

fig, ax = plt.subplots(figsize=(10,10))

#Make a scatter plot with success rate data
ax.scatter(x, y,)

#Adding labels and text
ax.set_xlabel('RICE', fontsize=14)
ax.set_ylabel('ICE', fontsize=14)
ax.set_title('RICE vs ICE', fontsize=18)
ax.text(.46, .39, 'low RICE and ICE', fontsize=10, alpha=.7)
ax.text( 90, 15, 'high RICE and ICE', fontsize=10, alpha=.7)
ax.text( 10, 15, 'high ICE and low RICE', fontsize=10, alpha=.7)
ax.text( 90, 1, 'high RICE and low ICE', fontsize=10, alpha=.7)

for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.04, i.get_height()+40,             str(round((i.get_height()), 2)), fontsize=11, color='#010fae', rotation=45)
for i, txt in enumerate(hypothesis.index.values):
    ax.annotate(txt, (x[i], y[i]), xytext=(5,5), textcoords='offset points',fontsize=11)


# **Conclusion**

# > Hypothesis priotization with 'RICE' framework has different priority compared to hypothesis priotization with 'ICE' framework. This is due to the 'Reach' score. For example, the two most priotized hypothesis on 'ICE' framework has lower 'Reach' score. This cause change in priority in two framework.

# ## Part 2. A/B Test Analysis
# 

# There are two dataframes. One gives information about orders taken and one gives the information about visits on online store.
# On both dataframes visitor/buyer is grouped in two groups. The conversion rate from visit to sucessful order and average purchase
# size of each buyer is analyzed for user from both group. The hypothesis is tested to find out is there significant difference in 
# conversion rate and average purchase size between 'group A' and 'group B'.

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment v.2</b> 
#     
# Thank you!
# 
# </div>

# In[13]:


orders= pd.read_csv('/datasets/orders_us.csv')
visits= pd.read_csv('/datasets/visits_us.csv')


# In[14]:


print(orders[orders['group']=='A'].sort_values('revenue', ascending= False).head())


# In[15]:


print(orders[orders['group']=='B'].sort_values('revenue', ascending= False).head())


# In[16]:


print(visits[visits['group']=='A'].sort_values('visits', ascending= False).head())


# In[17]:


print(visits[visits['group']=='B'].sort_values('visits', ascending= False).head())


# In[18]:


visits.isnull().sum()


# In[19]:


orders.isnull().sum()


# In[20]:


print(orders.duplicated().sum())


# In[21]:


print(visits.duplicated().sum())


# > no missing values and duplicated data

# In[22]:


visits['date'] = visits['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
orders['date'] = orders['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))


# ### Graph cumulative revenue by group. Make conclusions and conjectures.

# In[23]:


datesGroups = orders[['date','group']].drop_duplicates() 


# In[24]:


ordersAggregated = (datesGroups.apply(lambda x: orders[np.logical_and(orders['date'] <= x['date'], 
                    orders['group'] == x['group'])].agg({'date' : 'max', 'group' : 'max', 'transactionId' : pd.Series.nunique, 
                    'visitorId' : pd.Series.nunique, 'revenue' : 'sum'}), axis=1).sort_values(by=['date','group']) 
                   )


# In[25]:


visitorsAggregated = datesGroups.apply(lambda x: visits[np.logical_and(visits['date'] <= x['date'], visits['group'] == x['group'])].agg({'date' : 'max', 'group' : 'max', 'visits' : 'sum'}), axis=1).sort_values(by=['date','group']) 


# In[26]:


cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']


# In[27]:


print(cumulativeData.head())


# In[28]:


# DataFrame with cumulative orders and cumulative revenue by day, group A
cumulativeRevenueA = cumulativeData[cumulativeData['group']=='A'][['date','revenue', 'orders']]

# DataFrame with cumulative orders and cumulative revenue by day, group B
cumulativeRevenueB = cumulativeData[cumulativeData['group']=='B'][['date','revenue', 'orders']]

# Plotting the group A revenue graph 
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue'], label='A');

# Plotting the group B revenue graph 
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue'], label='B');
plt.xticks(rotation = 90);
plt.grid();

plt.legend(); 
plt.title('cumulative revenue ');




# Group B has higher cummulative  revenue compared to group A throughout the year. Significant difference can be seen after '2019-08-17'

# ### Graph cumulative average order size by group. Make conclusions and conjectures.

# In[29]:


plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue']/cumulativeRevenueA['orders'], label='A');
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue']/cumulativeRevenueB['orders'], label='B');
plt.xticks(rotation = 90);
plt.grid();
plt.legend(); 
plt.title('cumulative average order size ');


# In general group B has higher cumulative average order size. In average order size of  group B have surged three times and there is 
# genarally decreasing trend  in average order size after increase in average order size. For group A cumulative average order size has generally lower value 
# except around '2019-08-13'. This means there must be some big orders and outliers!

# ###  Graph the relative difference in cumulative average order size for group B compared with group A. Make conclusions and conjectures.
# 

# In[30]:


# gathering the data into one DataFrame
mergedCumulativeRevenue = cumulativeRevenueA.merge(cumulativeRevenueB, left_on='date', right_on='date', how='left', suffixes=['A', 'B'])

# plotting a relative difference graph for the average purchase sizes
plt.plot(mergedCumulativeRevenue['date'], (mergedCumulativeRevenue['revenueB']/mergedCumulativeRevenue['ordersB'])/(mergedCumulativeRevenue['revenueA']/mergedCumulativeRevenue['ordersA'])-1)

# adding the X axis
plt.axhline(y=0, color='black', linestyle='--');
plt.xticks(rotation = 90);
plt.grid();
plt.title('relative difference in cumulative average order size');


# Generally group B has higher average order size compared to group A expect at start of the period and at around 2019-08- 13. This means there must be some big orders and outliers!

# ###  Calculate each group's conversion rate as the ratio of orders to the number of visits for each day. Plot the daily conversion rates of the two groups and describe the difference. Draw conclusions and make conjectures.

# In[31]:


# calculating cumulative conversion
cumulativeData['conversion'] = cumulativeData['orders']/cumulativeData['visitors']

# selecting data on group A 
cumulativeDataA = cumulativeData[cumulativeData['group']=='A']

# selecting data on group B
cumulativeDataB = cumulativeData[cumulativeData['group']=='B']

# plotting the graphs
plt.plot(cumulativeDataA['date'], cumulativeDataA['conversion'], label='A')
plt.plot(cumulativeDataB['date'], cumulativeDataB['conversion'], label='B')
plt.legend()
plt.grid();

plt.xticks(rotation = 90);
plt.title('Conversion Rates');


# Conversion rate seems to have better in group B expect for few days before 2019-08-06. The spike in conversion rate at start should be due to some sort offers or discounts!

#  ### Plot a scatter chart of the number of orders per user. Make conclusions and conjectures.

# In[32]:


ordersByUsers = (
    orders.drop([ 'group','revenue', 'date'], axis=1)
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)

ordersByUsers.columns = ['userId', 'orders']
ordersByUsers.head()


# In[33]:


ordersByUsersA = orders[orders['group']=='A'].groupby('visitorId', as_index=False).agg({'transactionId' : pd.Series.nunique})
ordersByUsersA.columns = ['userId', 'orders']

ordersByUsersB = orders[orders['group']=='B'].groupby('visitorId', as_index=False).agg({'transactionId' : pd.Series.nunique})
ordersByUsersB.columns = ['userId', 'orders']


# In[34]:




x_values = pd.Series(range(0, len(ordersByUsersA['orders'])))
plt.scatter(x_values, ordersByUsersA['orders'], c='red') 

x_values = pd.Series(range(0, len(ordersByUsersB['orders'])))
plt.scatter(x_values, ordersByUsersB['orders'], c='blue')

plt.xlabel('users');
plt.ylabel('no of orders');
plt.title('Number of orders per user( blue--> group B, red--> group A)');


# By far highest propertion of users seems make one order. The second highest propertion of users make two orders follwed by users who made three orders.
# Remaining users seems to be in small proportion.

# ###  Calculate the 95th and 99th percentiles for the number of orders per user. Define the point at which a data point becomes an anomaly.
# 

# In[35]:


print('95th and 99th percentiles for the number of orders per user:  {}'.format(np.percentile(ordersByUsers['orders'], [ 95, 99]))) 


# Only 5 % of users make more than two orders and only 1 % of users make more than 4 orders.

#  ### Plot a scatter chart of order prices. Make conclusions and conjectures.
# 

# In[36]:


x_values = pd.Series(range(0,len(orders['revenue'])))
plt.figure(figsize=(16, 12), dpi=80)
plt.scatter(x_values, orders.revenue)
plt.xlabel('orders');
plt.ylabel('prices');
plt.title('Scatter chart of order prices');


# Due to very few outliers, the distribution of price  has not illustrated clearly. However there are very few order whose revenue is above 1000.
# 

# ### Calculate the 95th and 99th percentiles of order prices. Define the point at which a data point becomes an anomaly.
# 

# In[37]:


print('95th and 99th percentiles for prices of orders:  {}'.format(np.percentile(orders['revenue'], [ 95, 99])))


# 95 % of order has revenue below 436, and 99% of orders have revenue below 901.

# ### Find the statistical significance of the difference in conversion between the groups using the raw data. Make conclusions and conjectures.
# 

# **Null HYpothesis :**
#     
#     There is  not statically significant difference in convertion between group A and group B.
#     
# **Alternative HYpothesis :**
#     
#     There is  statically significant difference in convertion between group A and group B.

# In[38]:



sampleA = pd.concat([ordersByUsersA['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='A']['visits'].sum() - len(ordersByUsersA['orders'])), name='orders')],axis=0)

sampleB = pd.concat([ordersByUsersB['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='B']['visits'].sum() - len(ordersByUsersB['orders'])), name='orders')],axis=0)

print("{0:.3f}".format(stats.mannwhitneyu(sampleA, sampleB)[1]))

print("{0:.3f}".format(sampleB.mean()/sampleA.mean()-1)) 


# P- value is considerably lower than 0.05, so null hypothesis is rejected. The group conversion rate has statically significant difference.
# 
# The group conversion gain for group B has 13.8 % compared to group A.
# 

# ###  Find the statistical significance of the difference in average order size between the groups using the raw data. Make conclusions and conjectures.
# 

# **Null HYpothesis :**
#     
#     There is not statically significant difference in average order size between group A and group B.
#     
# **Alternative HYpothesis :**
#     
#     There is statically significant difference in average order size between group A and group B.

# In[39]:


print('{0:.3f}'.format(stats.mannwhitneyu(orders[orders['group']=='A']['revenue'], orders[orders['group']=='B']['revenue'])[1]))
print('{0:.3f}'.format(orders[orders['group']=='B']['revenue'].mean()/orders[orders['group']=='A']['revenue'].mean()-1)) 


# The p-value is considerably higher than 0.05, so there's no reason to reject the null hypothesis and conclude that average order size does not differs between the groups. Nonetheless, the average order size for group A is much smaller than it is for group B.
# The relative difference in average order size between two group is 25.2%. The difference appeared should be due to outliers.

# ### Find the statistical significance of the difference in conversion between the groups using the filtered data. Make conclusions and conjectures.

# **Null HYpothesis :**
#     
#     There is  not statically significant difference in convertion between group A and group B.
#     
# **Alternative HYpothesis :**
#     
#     There is  statically significant difference in convertion between group A and group B.

# In[40]:


orders_lim = np.percentile(ordersByUsers['orders'], 99 )
usersWithManyOrders = pd.concat([ordersByUsersA[ordersByUsersA['orders'] > orders_lim ]['userId'], ordersByUsersB[ordersByUsersB['orders'] > 4]['userId']], axis = 0)

orders1= orders
orders1.columns=['ransactionId', 'userId', 'date', 'revenue', 'group']
revenue_lim = np.percentile(orders1['revenue'], 99) 
usersWithExpensiveOrders = orders1[orders1['revenue'] > revenue_lim]['userId']
abnormalUsers = pd.concat([usersWithManyOrders, usersWithExpensiveOrders], axis = 0).drop_duplicates().sort_values()
print(abnormalUsers.head(5))
print(abnormalUsers.shape) 


# In[41]:


sampleAFiltered = pd.concat([ordersByUsersA[np.logical_not(ordersByUsersA['userId'].isin(abnormalUsers))]['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='A']['visits'].sum() - len(ordersByUsersA['orders'])),name='orders')],axis=0)

sampleBFiltered = pd.concat([ordersByUsersB[np.logical_not(ordersByUsersB['userId'].isin(abnormalUsers))]['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='B']['visits'].sum() - len(ordersByUsersB['orders'])),name='orders')],axis=0) 


# In[42]:


sampleAFiltered.head(3)


# In[43]:


print("{0:.3f}".format(stats.mannwhitneyu(sampleAFiltered, sampleBFiltered)[1]))
print("{0:.3f}".format(sampleBFiltered.mean()/sampleAFiltered.mean()-1)) 


# P- value is considerably lower than 0.05, so null hypothesis is rejected. The group conversion rate has statically significant difference.
# 
# The group conversion gain for group B has 15.3 % compared to group A.
# 

#  ### Find the statistical significance of the difference in average order size between the groups using the filtered data. Make conclusions and conjectures.
# 

# **Null HYpothesis :**
#     
#     There is not statically significant difference in average order size between group A and group B.
#     
# **Alternative HYpothesis :**
#     
#     There is statically significant difference in average order size between group A and group B.

# In[44]:


print('{0:.3f}'.format(stats.mannwhitneyu(
    orders[np.logical_and(
        orders['group']=='A',
        np.logical_not(orders['userId'].isin(abnormalUsers)))]['revenue'],
    orders[np.logical_and(
        orders['group']=='B',
        np.logical_not(orders['userId'].isin(abnormalUsers)))]['revenue'])[1]))

print('{0:.3f}'.format(
    orders[np.logical_and(orders['group']=='B',np.logical_not(orders['userId'].isin(abnormalUsers)))]['revenue'].mean()/
    orders[np.logical_and(
        orders['group']=='A',
        np.logical_not(orders['userId'].isin(abnormalUsers)))]['revenue'].mean() - 1)) 


# The p-value is considerably higher than 0.05, so there's no reason to reject the null hypothesis and conclude that average order
# size does not differs between the groups. Nonetheless, the average order size for group A is slightly higher than average order size  for group B.
# The relative difference in average order size between two group is 0.6%. 

# ### Make a decision based on the test results. The possible decisions are: Stop the test, consider one of the groups the leader.
# **Stop the test,** 
# 
# **conclude that there is no difference between the groups.** 
# 
# **Continue the test.**

# > At first, from the graph for cumulative revenue and average order size, the average order size for group B seemed to be significantly
# different (higher) than for group A. But, after removal of outliers, there is not strong base to conclude that group B has significantly
# different average size.

# > But, in the case of conversion rate, the strong evidence of having higher conversion rate for group B compared to group A are
# found from initial graph, manwhitney test for both raw data and filtered data (filtering outliers). So, the higher commulative revenue for group B can be understandable.

# >**The test can be stopped with the conclusion that the group B have higher conversion rate though the average purrchase size do not have significant difference.**

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment v.1</b> 
#     
# Correct! Thank you for your job here!
# </div>
