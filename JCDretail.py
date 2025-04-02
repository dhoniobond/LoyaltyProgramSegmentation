# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:48:46 2023

@author: danush
"""
#importing libraries
import pandas as pd
import numpy as np
#reading csv
df=pd.read_csv('customer-data.csv')

#descriptive statistics
df.head()
df.tail()
df.columns
df.dtypes
#changing data type of customer id as a string 
df['cust_id']=df['cust_id'].astype('str')
stats=df.describe()
df.isnull().sum()
#7038 null values found in cust_income hence ignoring the metric for clustering.
df['response_flag'].value_counts()

##########################################################################################
#importing seaborn for visualizations
import seaborn as sns
#comparing the categories that did and did not respond to the promotion against diffrent sales metrics
 
sns.boxplot(x='response_flag',y='basket_margin',data=df)
sns.boxplot(x='response_flag',y='total_dollars_L6M',data=df)
sns.boxplot(x='response_flag',y='total_num_purch_L6M',data=df)
sns.boxplot(x='response_flag',y='days_since_last_visit',data=df)

#creating subsets of the dataframe on the basis of response flag
df1=df[df['response_flag']==0]
df2=df[df['response_flag']==1]

#plotting sales metrics for individual subsets
sns.boxplot(x='response_flag',y='basket_margin',data=df1)
sns.boxplot(x='response_flag',y='basket_margin',data=df2)

sns.boxplot(x='response_flag',y='total_dollars_L6M',data=df2)

##########################################################################################

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#subsetting data with numeric values for clustering
df3=df[['cust_age','hhld_size','days_since_last_visit','months_since_first_purch','num_apparel_L6M','num_elec_L6M','num_haba_L6M','num_hw_L6M','total_num_purch_L6M','dollars_apparel_L6M','dollars_elec_L6M','dollars_haba_L6M','dollars_hw_L6M','total_dollars_L6M','basket_margin','margin_hw','margin_elec','margin_haba','margin_apparel']]
#df3=pd.get_dummies(df,drop_first=True)
#data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df3)
scaled_df=scaler.transform(df3)

#clustering import initialize train interpret
#randomly going with 4 clusters

km4= KMeans(n_clusters=4,random_state=0)#initialize
km4.fit(scaled_df)#training finding the clusters
km4.inertia_ 
#wcv and silk score
km4.inertia_ #wcv (withing cluster variation) 
silhouette_score(scaled_df, km4.labels_) ##silk score


wcv=[]

for i in range(2,18):
    km=KMeans(n_clusters=i,random_state=0)#initialize
    km.fit(scaled_df)#train
    wcv.append(km.inertia_)
   #cluster variation
    
#plotting elbow plot to decide value of k

plt.plot(range(2,18),wcv)
plt.xlabel('No of clusters')
plt.ylabel('within cluster variation')


#from the plot we can choose k as 6 as the curve flattens from that point


km6=KMeans(n_clusters=6, random_state=0)
km6.fit(scaled_df)
df['labels']= km6.labels_
df4= df.groupby('labels').mean()



km4= KMeans(n_clusters=4, random_state=0)
km4.fit(scaled_df)
df['labels']=km4.labels_
df4=df.groupby('labels').mean()


#we're thiking that 4 clusters may be better


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df3=df[['cust_age','hhld_size','days_since_last_visit','months_since_first_purch','num_apparel_L6M','num_elec_L6M','num_haba_L6M','num_hw_L6M','total_num_purch_L6M','dollars_apparel_L6M','dollars_elec_L6M','dollars_haba_L6M','dollars_hw_L6M','total_dollars_L6M','basket_margin','margin_hw','margin_elec','margin_haba','margin_apparel']]

#data scaling

scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df3)


linked = linkage(scaled_df, 'ward')#gets a n-1 *4 matrix
dendrogram(linked) #uses the matrix to get to draw the dendrogram
plt.title("Dendrogram")
plt.xlabel('Food')
plt.ylabel('euclidean')
plt.show()

hc=AgglomerativeClustering(n_clusters=3, linkage='ward')
hc.fit(scaled_df)

df['labels']=hc.labels_

df.groupby('labels').mean()



