#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 20:39:11 2023

@author: kaushiknarasimha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('/Users/kaushiknarasimha/Downloads/2016_2021/States_Fatalities.csv')

data.head(5)

data.info()

data['Licensed Drivers (Thousands)']=data['Licensed Drivers (Thousands)'].str.replace(',','')

data['Registered Vehicles (Thousands)']=data['Registered Vehicles (Thousands)'].str.replace(',','')

data['Population (Thousands)']=data['Population (Thousands)'].str.replace(',','')

data['Total Killed']=data['Total Killed'].str.replace(',','')

data.head(5)

data.info()

data.tail(5)

data['Licensed Drivers (Thousands)']=data['Licensed Drivers (Thousands)'].astype(int)

data['Registered Vehicles (Thousands)']=data['Registered Vehicles (Thousands)'].astype(int)

data['Population (Thousands)']=data['Population (Thousands)'].astype(int)

data['Total Killed']=data['Total Killed'].astype(int)

data.info()

# Extract the features you want to cluster on (e.g. latitude, longitude, hour of day)
X = data[['Fatalities per 100,000 Drivers','Fatalities per 100,000 Registered Vehicles','Fatalities per 100,000 Population']].values

X

# Calculate the sum of squared distances for different values of k
ssd = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    ssd.append(kmeans.inertia_)
    

# Plot the elbow curve
plt.plot(range(1, 11), ssd)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Distances')
plt.show()

from sklearn.metrics import silhouette_score


range_n_clusters = [3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
 
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
 
 # silhouette score
    silhouette_avg.append(silhouette_score(X, labels))
plt.plot(range_n_clusters,silhouette_avg,"bx-")
plt.xlabel("Values of K") 
plt.ylabel("Silhouette score") 
plt.title("Silhouette analysis For Optimal k")
plt.show()


from yellowbrick.cluster import SilhouetteVisualizer
 
fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [3, 4, 5, 6 ]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-2][mod],  title="custom title")
    visualizer.ax.set_xlabel("K = "+ str(i))
    visualizer.fit(X)
    

# apply K means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels2 = kmeans.labels_


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf["Cluster"] = labels2
import seaborn as sns
sns.scatterplot(data=principalDf, x="principal component 1", y="principal component 2", hue ="Cluster" )

# apply K means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels3 = kmeans.labels_


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf["Cluster"] = labels3
import seaborn as sns
sns.scatterplot(data=principalDf, x="principal component 1", y="principal component 2", hue ="Cluster" )

# apply K means clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels4 = kmeans.labels_



labels4

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf["Cluster"] = labels4
import seaborn as sns
sns.scatterplot(data=principalDf, x="principal component 1", y="principal component 2", hue ="Cluster" )

data['Cluster'] = labels4

data

data[data['Cluster'] == 0].State

data[data['Cluster'] == 1].State

data[data['Cluster'] == 2].State

data[data['Cluster'] == 3].State

# apply K means clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
labels5 = kmeans.labels_


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf["Cluster"] = labels5
import seaborn as sns
sns.scatterplot(data=principalDf, x="principal component 1", y="principal component 2", hue ="Cluster" )

import plotly.express as px


data1=pd.DataFrame(X)

data1.columns=['Fatalities by Drivers','Fatalities by Registered Vehicles','Fatalities by Population']

fig = px.scatter_3d(data1, x=data1["Fatalities by Drivers"], y=data1["Fatalities by Registered Vehicles"], z=data1["Fatalities by Population"],
              color=labels4)
fig.show()
fig.write_html("test4.html")

fig = px.scatter_3d(data1, x=data1["Fatalities by Drivers"], y=data1["Fatalities by Registered Vehicles"], z=data1["Fatalities by Population"],
              color=labels3)
fig.show()
fig.write_html("test3.html")

fig = px.scatter_3d(data1, x=data1["Fatalities by Drivers"], y=data1["Fatalities by Registered Vehicles"], z=data1["Fatalities by Population"],
              color=labels5)
fig.show()
fig.write_html("test5.html")


fig = px.scatter_3d(data1, x=data1["Fatalities by Drivers"], y=data1["Fatalities by Registered Vehicles"], z=data1["Fatalities by Population"],
              color=labels2)
fig.show()
fig.write_html("test2.html")




principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
pca_df=principalDf.copy()
pca_df.rename(columns = {'principal component 1':'PCA1', 'principal component 2':'PCA2'}, inplace = True)
pca_df["cluster"] = labels4



# apply K means clustering
kmeans_test = KMeans(n_clusters=4, random_state=0).fit(X)
labels4_test = kmeans.labels_




centroids = kmeans_test.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
## add to df
pca_df['cen_x'] = pca_df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3]})
pca_df['cen_y'] = pca_df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_x[3]})
# define and map colors
colors = ['#DF2020', '#81DF20', '#2095DF','#FFA500' ]
pca_df['c'] = pca_df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]})



from scipy.spatial import ConvexHull
import numpy as np
fig, ax = plt.subplots(1, figsize=(8,8))
# plot data
plt.scatter(pca_df.PCA1, pca_df.PCA2, c=pca_df.c, alpha = 0.6, s=10)
# plot centers
#plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)
# draw enclosure
for i in pca_df.cluster.unique():
    points = pca_df[pca_df.cluster == i][['PCA1', 'PCA2']].values
    # get convex hull
    hull = ConvexHull(points)
    # get x and y coordinates
    # repeat last point to close the polygon
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    # plot shape
    plt.fill(x_hull, y_hull, alpha=0.3, c=colors[i])
    
#plt.xlim(-10,10)
#plt.ylim(-10,10)


# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(x=pca_df.PCA1, y=pca_df.PCA2, color=pca_df.cluster, title="K = 4")
fig.show()
fig.write_html('k_4_plot')



# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(pca_df, x="PCA1", y="PCA2", color="c",title="K = 4")
fig.show()
fig.write_html('k_4_plot_1')




from scipy import interpolate
fig, ax = plt.subplots(1, figsize=(6,6))
plt.scatter(pca_df.PCA1, pca_df.PCA2, c=pca_df.c, alpha = 0.6, s=10)
    
for i in pca_df.cluster.unique():
    # get the convex hull
    points = pca_df[pca_df.cluster == i][['PCA1', 'PCA2']].values
    hull = ConvexHull(points)
    # get x and y coordinates
    # repeat last point to close the polygon
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])

    
    # interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], 
                                    u=dist_along, s=0, per=1)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    plt.fill(interp_x, interp_y, '--', c=colors[i], alpha=0.2)
    
    
    
