# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:18:39 2018

@author: Leonardo
"""

# Speed Analysis V3.0
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model

def speed_analysisv2(df):
    [a,b,c,centroids]=clustering_kmeans(df)
    #[a,b,c,centroids]=clustering_mini(df)
    #[a,b,c,centroids]=clustering_agglo(df)
    f_data4=linearRegression(a,b,c)
    
    # SVM 
    with open('filename2.pkl', 'rb') as f:
        clf = pickle.load(f)
        a=clf.predict(f_data4)
        if min(centroids[:,1])<=0.0001 and (a==2 or a==3):
            a=1
        if a==0:
            speed_change="aumenta"
        if a==1:
            speed_change="detiene"
        if a==2:
            speed_change="disminuye"
        if a==3:
            speed_change="igual"
            
    return speed_change

def clustering_kmeans(df):
    kmeans=KMeans(n_clusters=3,random_state=0).fit(df)
    centroids = kmeans.cluster_centers_
    labels=kmeans.labels_
    
    # Ordered slopes
    m1 = np.mean(df[labels==0,0])
    m2 = np.mean(df[labels==1,0])
    m3 = np.mean(df[labels==2,0])

    m=np.array([m1,m2,m3])
    ms=np.argsort(m)


    a = df[labels==ms[0]]
    b = df[labels==ms[1]]
    c = df[labels==ms[2]]
    
    return a,b,c,centroids

def clustering_mini(df):
    
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=45,
                      n_init=10, max_no_improvement=10, verbose=0)
    # Model clustering
    mbk.fit(df)

    centroids = mbk.cluster_centers_
    labels=mbk.labels_
    
    # Ordered slopes
    m1 = np.mean(df[labels==0,0])
    m2 = np.mean(df[labels==1,0])
    m3 = np.mean(df[labels==2,0])

    m=np.array([m1,m2,m3])
    ms=np.argsort(m)


    a = df[labels==ms[0]]
    b = df[labels==ms[1]]
    c = df[labels==ms[2]]
    
    return a,b,c,centroids

def clustering_agglo(df):
    
    model = AgglomerativeClustering(n_clusters=3,
                                    linkage="complete", affinity="euclidean")
       
    # Model clustering
    model.fit(df)

    labels=model.labels_
    
    # Ordered slopes
    m1 = np.mean(df[labels==0,0])
    m2 = np.mean(df[labels==1,0])
    m3 = np.mean(df[labels==2,0])
    
    # Calculated centroids
    m11 = np.mean(df[labels==0,1])
    m21 = np.mean(df[labels==1,1])
    m31 = np.mean(df[labels==2,1])

    centroids = np.array([[m1,m11],[m2,m21],[m3,m31]])


    m=np.array([m1,m2,m3])
    ms=np.argsort(m)


    a = df[labels==ms[0]]
    b = df[labels==ms[1]]
    c = df[labels==ms[2]]
    
    return a,b,c,centroids



def linearRegression(a,b,c):
    
    X_train1=np.array(a[:,0]).reshape(-1,1)
    Y_train1=a[:,1]
    X_train2=np.array(b[:,0]).reshape(-1,1)
    Y_train2=b[:,1]
    X_train3=np.array(c[:,0]).reshape(-1,1)
    Y_train3=c[:,1]

    # Linear Regression Object
    regr = linear_model.LinearRegression()
    regr1 = linear_model.LinearRegression()
    regr2 = linear_model.LinearRegression()

    # Model training
    regr.fit(X_train1, Y_train1)
    regr1.fit(X_train2, Y_train2)
    regr2.fit(X_train3, Y_train3)



    # Data Classification
    f_data4=np.array([regr.coef_,regr1.coef_,regr2.coef_]).reshape(1,-1)
    
    return f_data4


if __name__ == '__main__':
	main()