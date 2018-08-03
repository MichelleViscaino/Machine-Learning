# Confussion Matrix
# Speed Analysis V3.0 y V4.0
# Michelle Viscaino

import pickle
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn import linear_model

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



pred=[]

def main():
    # Save predictions
    global pred
    
    file = r'MatrixConfussion.xlsx' # Load file
    df = pd.read_excel(file,index_col=None, header=None)
    df = df.values 
    # create a time vector
    t=np.linspace(0,1,len(df))
    
    for i in range(len(df[0])):
        spd=df[:,i]
        # Concatenate matrix
        df1=np.vstack([t,spd])
        df1=np.transpose(df1)
        [a,b,c,centroids]=clusteringD(df1)
        f_data4=linearRegression(a,b,c)
        with open('filename2.pkl', 'rb') as f:
            clf = pickle.load(f)
            a=clf.predict(f_data4)
            pred.append(a)
    

    file1 =pd.read_excel(r'RealMatrix.xlsx', index_col=None, header=None)
    act = file1.values
    act=np.asarray(act)
    
    # Compute confussion matrix
    cnf_matrix = confusion_matrix(act, pred)
    np.set_printoptions(precision=2)
    # Plot confussion matrix
    class_names=['Aumenta','Detiene','Disminuye','Igual']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confussion matrix')


def plot_confusion_matrix(cm, classes,
                         title='Confussion matrix',
                          cmap=plt.cm.PuBuGn):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    


def clusteringD(df):
    
    #K-Means Clustering
    kmeans=KMeans(n_clusters=3,random_state=0).fit(df)   
    centroids = kmeans.cluster_centers_
    labels=kmeans.labels_
    
    m1 = np.mean(df[labels==0,0])
    m2 = np.mean(df[labels==1,0])
    m3 = np.mean(df[labels==2,0])

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